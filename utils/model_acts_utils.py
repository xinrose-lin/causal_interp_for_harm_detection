import torch as t
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm 

## load pythia with hooked transformer for consistent comparison

device = "cpu"
pythia_model: HookedSAETransformer = HookedSAETransformer.from_pretrained('EleutherAI/pythia-70m-deduped', device=device)

sae_release = "pythia-70m-deduped-res-sm"
sae_id = "blocks.4.hook_resid_post"
sae = SAE.from_pretrained(sae_release, sae_id)[0]

 ## gather hidden layer rep of prompts
def get_acts(concept_prompts, 
           model):
    output, cache = model.run_with_cache(concept_prompts)
    acts = cache["blocks.4.hook_resid_post"]
    return acts 

## gather sparse rep of prompts 
def get_sparse_acts(concept_prompts, model): 
    output, cache = model.run_with_cache_with_saes(
            concept_prompts,
            saes=[sae],
            stop_at_layer= sae.cfg.hook_layer + 1,
        )
    sparse_acts = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"]

    return sparse_acts

## used to compute casual distribution 
## to locate most relevant concept latents
def get_ablation_effects(acts, probe):

    # gather ablation effect for nonzero latents
    active_latents = len(acts.mean(dim=(1)).nonzero()[:, 1].unique())

    ablation_effects = t.zeros(sae.cfg.d_sae)

    # reference point 
    full_acts = acts.mean(dim=1)
    logits = probe(full_acts)

    # compute causal score for each latent
    for latent_idx in tqdm(range(1, active_latents)): 
        ablated_acts = full_acts.clone()

        # ablate latent
        ablated_acts[:, latent_idx] = 0.0
        # compute logits_with_ablation
        logits_with_ablation = probe(ablated_acts)

        ablation_effects[latent_idx] = (logits - logits_with_ablation).mean() # over prompts

    return ablation_effects
