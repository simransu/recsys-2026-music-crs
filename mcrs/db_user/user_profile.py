import os
import json
import random
from datasets import load_dataset, concatenate_datasets
from mcrs.db_item.music_catalog import normalize_entity_id

class UserProfileDB:
    def __init__(self,
            dataset_name: str,
            split_types: list[str],
        ):
        metadata_dataset = load_dataset(dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in split_types])
        self.default_columns = ['user_id', 'age_group', 'gender', 'country_name']
        self.user_profiles = {normalize_entity_id(item["user_id"]): item for item in metadata_concat_dataset}

    def id_to_profile(self, user_id: str):
        user_profile = self.user_profiles[normalize_entity_id(user_id)]
        return user_profile

    def id_to_profile_str(self, user_id: str):
        user_profile = self.user_profiles[normalize_entity_id(user_id)]
        profile_str = [f"{key}: {user_profile[key]}" for key in self.default_columns]
        return "\n".join(profile_str)
