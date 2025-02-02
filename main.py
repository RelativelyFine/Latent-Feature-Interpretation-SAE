#!/usr/bin/env python3
import argparse
import torch
from sae_lens import SAE, HookedSAETransformer

def get_dashboard_html(
    sae_release: str = "gpt2-small",
    sae_id: str = "7-res-jb",
    feature_idx: int = 0,
    width: int = 1200,
    height: int = 600
) -> str:
    html_template = (
        "https://neuronpedia.org/{}/{}/{}?"
        "embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    )
    return (
        f'<iframe src="{html_template.format(sae_release, sae_id, feature_idx)}" '
        f'width="{width}" height="{height}"></iframe>'
    )

def load_prompts_from_file(filepath: str) -> list[str]:
    """Load prompts from a file, one per line, stripping whitespace."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines

def main():
    parser = argparse.ArgumentParser(description="Filter SAE features by positive/negative prompt sets.")
    parser.add_argument("--pos_file", type=str, required=True, help="Path to text file with positive prompts (one per line).")
    parser.add_argument("--neg_file", type=str, required=True, help="Path to text file with negative prompts (one per line).")
    parser.add_argument("--pos_threshold", type=float, default=0.1,
                        help="Feature must exceed this min activation across positive tokens.")
    parser.add_argument("--neg_threshold", type=float, default=0.1,
                        help="Feature must remain below this max activation across negative tokens.")
    parser.add_argument("--sae_release", type=str, default="gpt2-small-res-jb",
                        help="The SAE release to load, e.g. 'gpt2-small-res-jb'.")
    parser.add_argument("--sae_id", type=str, default="blocks.7.hook_resid_pre",
                        help="Which SAE to load (corresponding to a hook name).")
    parser.add_argument("--output_html", type=str, default="output.html",
                        help="Filename for the HTML output.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --------------------------------------------------------
    # 1. Load the GPT-2 model and the specified SAE
    # --------------------------------------------------------
    print("Loading model ...")
    model = HookedSAETransformer.from_pretrained("gpt2-small", device=device)

    print(f"Loading SAE: release={args.sae_release}, id={args.sae_id} ...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=device,
    )

    # --------------------------------------------------------
    # 2. Load prompts from files
    # --------------------------------------------------------
    pos_prompts = load_prompts_from_file(args.pos_file)
    neg_prompts = load_prompts_from_file(args.neg_file)
    print(f"Number of positive prompts: {len(pos_prompts)}")
    print(f"Number of negative prompts: {len(neg_prompts)}")

    # --------------------------------------------------------
    # 3. Compute activations for each set
    #    We'll track SAE activations for all tokens.
    # --------------------------------------------------------
    def get_all_token_acts(prompt: str) -> torch.Tensor:
        """
        Takes a single prompt, returns SAE activations
        for all token positions (shape: [seq_len, d_sae]).
        """
        tokens = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)
        _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
        sae_acts = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"][0]  # [seq_len, d_sae]
        return sae_acts  # All tokens

    print("Computing positive prompt activations ...")
    pos_acts_list = [get_all_token_acts(p) for p in pos_prompts]
    pos_acts = torch.cat([acts.reshape(-1, sae.cfg.d_sae) for acts in pos_acts_list], dim=0)
    print(f"Shape of pos_acts: {pos_acts.shape}")

    print("Computing negative prompt activations ...")
    neg_acts_list = [get_all_token_acts(p) for p in neg_prompts]
    neg_acts = torch.cat([acts.reshape(-1, sae.cfg.d_sae) for acts in neg_acts_list], dim=0)
    print(f"Shape of neg_acts: {neg_acts.shape}")

    # --------------------------------------------------------
    # 4. Filter for features based on activations across all tokens
    # --------------------------------------------------------
    print("Filtering features ...")

    # Compute mean activations across all tokens
    pos_mean = pos_acts.mean(dim=0)  # Shape: [d_sae]
    neg_mean = neg_acts.mean(dim=0)  # Shape: [d_sae]

    pos_mask = pos_mean > args.pos_threshold
    neg_mask = neg_mean < args.neg_threshold
    combined_mask = pos_mask & neg_mask

    # IDs (indices) of features passing the filter
    passing_features = torch.arange(sae.cfg.d_sae, device=pos_mean.device)[combined_mask]

    # Sort features by descending pos_mean
    sorted_indices = torch.argsort(pos_mean[passing_features], descending=True)
    passing_features = passing_features[sorted_indices]
    passing_features = passing_features[:10]

    print(f"Found {len(passing_features)} features passing filter.")
    if len(passing_features) == 0:
        print("No features meet criteria. Exiting.")
        return

    # --------------------------------------------------------
    # 5. Create an HTML file listing and embedding these features
    # --------------------------------------------------------
    print(f"Writing results to {args.output_html} ...")
    with open(args.output_html, "w", encoding="utf-8") as f:
        f.write("<html><body>\n")
        f.write("<h1>SAE Feature Filter Results</h1>\n")
        f.write(f"<p>Positive threshold: {args.pos_threshold}, Negative threshold: {args.neg_threshold}</p>\n")
        f.write(f"<h3>{len(passing_features)} Features Found</h3>\n")
        f.write("<table border='1' style='border-collapse: collapse;'>\n")
        f.write("<tr><th>Rank</th><th>Feature ID</th><th>pos_mean</th><th>neg_mean</th><th>Dashboard</th></tr>\n")
        for rank, feature_idx in enumerate(passing_features.tolist(), start=1):
            feature_pos_mean = pos_mean[feature_idx].item()
            feature_neg_mean = neg_mean[feature_idx].item()
            dashboard_iframe = get_dashboard_html(
                sae_release="gpt2-small",  # or args.sae_release
                sae_id="7-res-jb",        # or args.sae_id
                feature_idx=feature_idx,
                width=800,
                height=450
            )
            f.write("<tr>")
            f.write(f"<td>{rank}</td>")
            f.write(f"<td>{feature_idx}</td>")
            f.write(f"<td>{feature_pos_mean:.4f}</td>")
            f.write(f"<td>{feature_neg_mean:.4f}</td>")
            f.write(f"<td>{dashboard_iframe}</td>")
            f.write("</tr>\n")
        f.write("</table>\n")
        f.write("</body></html>\n")

    print(f"Done! See '{args.output_html}' for embedded dashboards.")

if __name__ == "__main__":
    main()