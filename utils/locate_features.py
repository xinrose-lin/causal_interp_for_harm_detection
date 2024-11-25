import torch as t
from sae_lens import SAE, HookedSAETransformer


def get_acts(concept_prompts, 
            sae_release = "pythia-70m-deduped-res-sm", 
            acts_id = "blocks.4.hook_resid_post", 
            device = "cuda" if t.cuda.is_available() else "cpu", 
                    save = None):
    # load SAE of specific layer, and model 
    model: HookedSAETransformer = HookedSAETransformer.from_pretrained('EleutherAI/pythia-70m-deduped', device=device)

    ## gather sparse rep of harmful prompts 
    output, cache = model.run_with_cache(concept_prompts)
    acts = cache[acts_id]
    return acts 

def get_sparse_acts(concept_prompts, 
                    sae_release = "pythia-70m-deduped-res-sm", 
                    sae_id = "blocks.4.hook_resid_post", 
                    device = "cuda" if t.cuda.is_available() else "cpu", 
                    save = None):
    """
    Input: 
        - batch of prompts (target concept prompts)
        - model: release (ref api of sae_lens: eg. pythia 70m)
        - target layer: sae_id (ref api of sae_lens eg. layer 4 output)

    Output: 
        - return sparse latent representations
        - assert len(sae_dimensions)

    Used for generating activations for: 
        - harmful_train_ds = load_sample_ds(sample_data="harmful_train_ds")
        
    """
    # Assert that the list is not empty and all elements are non-empty strings
    assert concept_prompts, "Error: concept_prompts cannot be empty"
    assert all(isinstance(prompt, str) and prompt for prompt in concept_prompts), "Error: All elements in concept_prompts must be non-empty strings"

    # load SAE of specific layer, and model 
    sae = SAE.from_pretrained(sae_release, sae_id)[0]
    model: HookedSAETransformer = HookedSAETransformer.from_pretrained('EleutherAI/pythia-70m-deduped', device=device)

    ## gather sparse rep of harmful prompts 
    output, cache = model.run_with_cache_with_saes(
            concept_prompts,
            saes=[sae],
            stop_at_layer= sae.cfg.hook_layer + 1,
        )
    sparse_acts = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"]

    return sparse_acts