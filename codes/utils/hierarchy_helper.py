import json
import torch
import os
import sys

class HierarchyHelper:
    def __init__(self, dataset, root_dir='codes/dataset'):
        """
        Helper Class: Load semantic hierarchy mapping relations (JSON) and convert them to Tensor indices.
        
        [STRICT MODE]: 
        1. If mapping files are missing -> CRASH (FileNotFoundError).
        2. If a specific word is missing in the mapping -> CRASH (ValueError).
        NO dummy data, NO default values.
        """
        self.dataset = dataset
        self.root = root_dir
        
        # 1. Load coarse-grained mapping JSON files (STRICT LOAD)
        self.verb_map_raw = self._load_json('verb_mapping.json')
        self.obj_map_raw = self._load_json('object_mapping.json')

        # 2. Get basic fine-grained lists from Dataset
        self.verbs = dataset.attrs
        self.objs = dataset.objs
        self.pairs = dataset.pairs 

        # [CRITICAL FIX] Use dataset's internal mapping, NOT enumerate order
        # Ensure that the indices we generate align perfectly with DataLoader labels
        self.attr2idx = dataset.attr2idx
        self.obj2idx = dataset.obj2idx

        # 3. Build coarse-grained parent category lists
        # Sort to ensure deterministic index order
        self.coarse_verbs = sorted(list(set(self.verb_map_raw.values())))
        self.coarse_objs = sorted(list(set(self.obj_map_raw.values())))

        print(f"[HierarchyHelper] Loaded {len(self.coarse_verbs)} coarse verbs and {len(self.coarse_objs)} coarse objects.")

        # 4. Build index mapping Tensors (STRICT BUILD)
        # v2cv: Map Fine-Verb-ID -> Coarse-Verb-ID
        self.v2cv_idx = self._build_mapping(self.verbs, self.verb_map_raw, self.coarse_verbs, self.attr2idx, "Verb")
        # o2co: Map Fine-Obj-ID -> Coarse-Obj-ID
        self.o2co_idx = self._build_mapping(self.objs, self.obj_map_raw, self.coarse_objs, self.obj2idx, "Object")
        
        # p2v, p2o: Map Pair-ID -> Verb-ID / Obj-ID
        self.p2v_idx, self.p2o_idx = self._build_pair_mapping()

    def _load_json(self, filename):
        path = os.path.join(self.root, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[HierarchyHelper] FATAL: Mapping file not found at {path}.")
        
        with open(path, 'r') as f:
            return json.load(f)

    def _build_mapping(self, child_list, raw_map, parent_list, child2idx, tag="Item"):
        """
        Strictly maps child concepts to parent indices.
        NOTE: Returns a Tensor where index `i` corresponds to the child with ID `i`.
        This requires creating a tensor of size [MaxID + 1] and filling it based on child2idx.
        """
        # Determine size of the mapping tensor (Max ID + 1)
        # Usually len(child_list) is enough if IDs are 0..N-1, but strictly we use max index.
        max_id = max(child2idx.values())
        mapping = torch.zeros(max_id + 1, dtype=torch.long)
        
        p_to_id = {name: i for i, name in enumerate(parent_list)}
        
        missing_items = []

        for child in child_list:
            # Get the CORRECT ID from dataset
            if child not in child2idx:
                continue # Should not happen if child_list comes from dataset
            
            c_id = child2idx[child]
            
            if child in raw_map:
                p_name = raw_map[child]
                if p_name in p_to_id:
                    mapping[c_id] = p_to_id[p_name]
                else:
                    raise ValueError(f"[HierarchyHelper] Logic Error: Parent '{p_name}' for '{child}' not found.")
            else:
                missing_items.append(child)
        
        if len(missing_items) > 0:
            print(f"[HierarchyHelper] Missing Mappings for {len(missing_items)} {tag}s:")
            print(missing_items[:10])
            raise ValueError(f"[HierarchyHelper] CRITICAL: Mapping JSON incomplete for {tag}.")
            
        return mapping

    def _build_pair_mapping(self):
        """
        Build Composition (Pair) to Verb and Object index mappings.
        """
        # [CRITICAL FIX] Use dataset's dictionary, DO NOT re-enumerate
        p2v = torch.zeros(len(self.pairs), dtype=torch.long)
        p2o = torch.zeros(len(self.pairs), dtype=torch.long)
        
        for i, (v, o) in enumerate(self.pairs):
            # i is the pair index (Assuming dataset.pairs order matches batch_target index)
            # This assumption is generally true for CompositionDataset.
            
            # Map Verb string -> Verb ID (aligned with DataLoader)
            p2v[i] = self.attr2idx[v]
            # Map Object string -> Object ID
            p2o[i] = self.obj2idx[o]
            
        return p2v, p2o

    def get_coarse_info(self):
        return self.coarse_verbs, self.coarse_objs

    def get_mappings(self):
        return self.v2cv_idx, self.o2co_idx, self.p2v_idx, self.p2o_idx