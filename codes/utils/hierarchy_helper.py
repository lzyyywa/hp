import json
import torch
import os

class HierarchyHelper:
    def __init__(self, dataset, root_dir='codes/dataset'):
        """
        Helper Class: Load semantic hierarchy mapping relations (JSON) and convert them to Tensor indices.
        
        Args:
            dataset: Instance of CompositionVideoDataset (requires access to .attrs, .objs, .pairs)
            root_dir: Directory storing mapping json files (default: codes/dataset)
        """
        self.dataset = dataset
        self.root = root_dir
        
        # 1. Load coarse-grained mapping JSON files
        # Note: Uses the specified filenames object_mapping.json and verb_mapping.json
        self.verb_map_raw = self._load_json('verb_mapping.json')
        self.obj_map_raw = self._load_json('object_mapping.json')

        # 2. Get basic fine-grained lists from Dataset
        # attrs -> verbs, objs -> objects
        self.verbs = dataset.attrs
        self.objs = dataset.objs
        
        # pairs -> list of (verb, object) tuples
        # Contains all combinations to be considered
        self.pairs = dataset.pairs 

        # 3. Build coarse-grained parent category lists (deduplicated and sorted as parent vocabularies)
        self.coarse_verbs = sorted(list(set(self.verb_map_raw.values())))
        self.coarse_objs = sorted(list(set(self.obj_map_raw.values())))

        print(f"[HierarchyHelper] Loaded {len(self.coarse_verbs)} coarse verbs and {len(self.coarse_objs)} coarse objects.")

        # 4. Build index mapping Tensors (core function)
        
        # (A) Original hierarchy: fine-grained -> coarse-grained
        # Tensor shape: [Num_Verbs], Content: corresponding Coarse Verb Index
        self.v2cv_idx = self._build_mapping(self.verbs, self.verb_map_raw, self.coarse_verbs, "Verb")
        
        # Tensor shape: [Num_Objects], Content: corresponding Coarse Object Index
        self.o2co_idx = self._build_mapping(self.objs, self.obj_map_raw, self.coarse_objs, "Object")
        
        # (B) Composition hierarchy: Composition (Pair) -> Verb / Object
        # Tensor shape: [Num_Pairs], Content: corresponding Verb/Object Index
        self.p2v_idx, self.p2o_idx = self._build_pair_mapping()

    def _load_json(self, filename):
        path = os.path.join(self.root, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"[HierarchyHelper] Error: Mapping file not found at {path}")
        
        with open(path, 'r') as f:
            return json.load(f)

    def _build_mapping(self, child_list, raw_map, parent_list, tag="Item"):
        """
        Universal construction function: Build Child Index -> Parent Index mapping Tensor
        """
        mapping = torch.zeros(len(child_list), dtype=torch.long)
        
        # Build parent name to index lookup table
        p_to_id = {name: i for i, name in enumerate(parent_list)}
        
        missing_count = 0
        for i, child in enumerate(child_list):
            if child in raw_map:
                p_name = raw_map[child]
                mapping[i] = p_to_id[p_name]
            else:
                # Fault tolerance: Map to index 0 by default if the term is not in JSON
                mapping[i] = 0
                missing_count += 1
                # Print only the first few missing items to avoid log spamming
                if missing_count <= 3:
                    print(f"[Warning] {tag} '{child}' not found in mapping json! Defaulting to parent index 0.")
        
        if missing_count > 0:
            print(f"[Warning] Total {missing_count} {tag}s missing from mapping json.")
            
        return mapping

    def _build_pair_mapping(self):
        """
        Build Composition (Pair) to Verb and Object index mappings.
        Used to implement hierarchical constraints of Composition ∈ Verb and Composition ∈ Object.
        """
        # Build Verb/Object name to index lookup tables
        verb_to_id = {v: i for i, v in enumerate(self.verbs)}
        obj_to_id = {o: i for i, o in enumerate(self.objs)}
        
        p2v = torch.zeros(len(self.pairs), dtype=torch.long)
        p2o = torch.zeros(len(self.pairs), dtype=torch.long)
        
        for i, (v, o) in enumerate(self.pairs):
            p2v[i] = verb_to_id[v]
            p2o[i] = obj_to_id[o]
            
        return p2v, p2o

    def get_coarse_info(self):
        """
        Return text lists of coarse-grained parent categories, 
        used for encoding with CLIP Text Encoder in subsequent steps.
        
        Returns:
            coarse_verbs (List[str])
            coarse_objs (List[str])
        """
        return self.coarse_verbs, self.coarse_objs

    def get_mappings(self):
        """
        Return all constructed index mapping Tensors.
        These Tensors will be moved to GPU for Loss calculation.
        
        Returns:
            v2cv_idx: [Num_Verbs] -> Coarse Verb Index
            o2co_idx: [Num_Objects] -> Coarse Object Index
            p2v_idx:  [Num_Pairs] -> Verb Index
            p2o_idx:  [Num_Pairs] -> Object Index
        """
        return self.v2cv_idx, self.o2co_idx, self.p2v_idx, self.p2o_idx