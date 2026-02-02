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
        # Ensure your dataset class exposes these attributes!
        self.verbs = dataset.attrs
        self.objs = dataset.objs
        self.pairs = dataset.pairs 

        # 3. Build coarse-grained parent category lists
        # Sort to ensure deterministic index order
        self.coarse_verbs = sorted(list(set(self.verb_map_raw.values())))
        self.coarse_objs = sorted(list(set(self.obj_map_raw.values())))

        print(f"[HierarchyHelper] Loaded {len(self.coarse_verbs)} coarse verbs and {len(self.coarse_objs)} coarse objects.")

        # 4. Build index mapping Tensors (STRICT BUILD)
        # v2cv: Map Fine-Verb-ID -> Coarse-Verb-ID
        self.v2cv_idx = self._build_mapping(self.verbs, self.verb_map_raw, self.coarse_verbs, "Verb")
        # o2co: Map Fine-Obj-ID -> Coarse-Obj-ID
        self.o2co_idx = self._build_mapping(self.objs, self.obj_map_raw, self.coarse_objs, "Object")
        
        # p2v, p2o: Map Pair-ID -> Verb-ID / Obj-ID
        self.p2v_idx, self.p2o_idx = self._build_pair_mapping()

    def _load_json(self, filename):
        path = os.path.join(self.root, filename)
        if not os.path.exists(path):
            # Must raise error! Prevent auto-generation logic from being reactivated
            raise FileNotFoundError(f"[HierarchyHelper] FATAL: Mapping file not found at {path}. Please provide valid mapping files.")
        
        with open(path, 'r') as f:
            return json.load(f)

    def _build_mapping(self, child_list, raw_map, parent_list, tag="Item"):
        """
        Strictly maps child concepts to parent indices.
        Raises ValueError if ANY child concept is missing from the mapping.
        """
        mapping = torch.zeros(len(child_list), dtype=torch.long)
        p_to_id = {name: i for i, name in enumerate(parent_list)}
        
        missing_items = []

        for i, child in enumerate(child_list):
            if child in raw_map:
                p_name = raw_map[child]
                if p_name in p_to_id:
                    mapping[i] = p_to_id[p_name]
                else:
                    # Parent name in JSON is not in the parent list? (Should be impossible if logic holds, but verify)
                    raise ValueError(f"[HierarchyHelper] Logic Error: Parent '{p_name}' for item '{child}' not found in parent vocabulary.")
            else:
                # Record missing items and raise unified error later
                missing_items.append(child)
        
        if len(missing_items) > 0:
            # Found words in dataset but not in JSON -> Raise error immediately!
            # Never map to 0 by default to prevent data contamination
            print(f"[HierarchyHelper] Missing Mappings for {len(missing_items)} {tag}s:")
            print(missing_items[:10]) # Print first 10
            raise ValueError(f"[HierarchyHelper] CRITICAL: The provided {tag} mapping JSON is incomplete. Missing {len(missing_items)} keys. Training aborted to prevent bad data.")
            
        return mapping

    def _build_pair_mapping(self):
        """
        Build Composition (Pair) to Verb and Object index mappings.
        Used for L_DA (summing independent logits) and L_TE (checking composition entailment).
        """
        verb_to_id = {v: i for i, v in enumerate(self.verbs)}
        obj_to_id = {o: i for i, o in enumerate(self.objs)}
        
        p2v = torch.zeros(len(self.pairs), dtype=torch.long)
        p2o = torch.zeros(len(self.pairs), dtype=torch.long)
        
        for i, (v, o) in enumerate(self.pairs):
            p2v[i] = verb_to_id[v]
            p2o[i] = obj_to_id[o]
            
        return p2v, p2o

    def get_coarse_info(self):
        """Returns the lists of coarse concepts for Prompt Learner initialization"""
        return self.coarse_verbs, self.coarse_objs

    def get_mappings(self):
        """Returns the Index Tensors for Loss calculation"""
        return self.v2cv_idx, self.o2co_idx, self.p2v_idx, self.p2o_idx