import os
import torch
import json
from datasets import load_dataset, concatenate_datasets

def format_metadata_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    # Handle stringified lists like "['What They Do']"
    s = str(value)
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return ", ".join(str(item) for item in parsed)
        except Exception:
            pass
    return s


def normalize_entity_id(value):
    return str(value).strip() if value is not None else ""

class MusicCatalogDB:
    def __init__(self,
            dataset_name: str,
            split_types: list[str],
            corpus_types: list[str],
        ):
        metadata_dataset = load_dataset(dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in split_types])
        self.corpus_types = corpus_types
        self.metadata_dict = {normalize_entity_id(item["track_id"]): item for item in metadata_concat_dataset}

    def id_to_metadata(self, track_id: str, use_semantic_id: bool = False):
        metadata = self.metadata_dict[normalize_entity_id(track_id)]
        track_id = metadata['track_id']
        entity_str = f"track_id: {track_id}"
        for corpus_type in self.corpus_types:
            corpus_type_value = format_metadata_value(metadata.get(corpus_type, "")).lower()
            entity_str += f", {corpus_type}: {corpus_type_value}"
        return entity_str
