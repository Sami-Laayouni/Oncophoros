"""
This script first looks for promoter sequences in a provided FASTA file that match gene names listed in a text file.
For each gene with a matching promoter, it constructs a CHOMP-style genetic circuit by adding pre-defined DNA modules.
It then generates an SBOL-like diagram of the circuit and saves it as both SVG and PNG files. 

This is basically a cool automation genetic circuit design tool for synthetic biology applications.
"""

import re
import os
import csv
from dna_features_viewer import GraphicFeature, GraphicRecord


CONFIG = {
    "PROMOTERS_FASTA": r"C:\Users\Admin\Desktop\IGEM Hackathon\data\Promoters.fasta",
    "GENES_FILE":      r"C:\Users\Admin\Desktop\IGEM Hackathon\data\genes.txt",
    "OUTDIR": "generated_genetic_circuits"
}

MODULES = {
    # ITR_5: Adeno-associated virus (AAV) 5' inverted terminal repeat sequence
    # Source: Earley et al., 2020; GenBank accession NC_001401.2 
    # (https://pmc.ncbi.nlm.nih.gov/articles/PMC7047122/)
    "ITR_5": "GAGCGGCCTCAGTGAGCGAGCGGGACAAAGTGGAGGTGGTGCTGCGAGGCAGGCGGGGTGAGAGACAGGCGCCGCCAGCGCCCGGCGGAAGCGCTGAGCGAGGCCAGCGAGGCGGACACGCGCCCCGCCGAGGA",

    "Promoter": None,  

    # Found on several sources
    "Kozak_Start": "GCCACCATGG",

    # Sensor_Protease: TEV protease cleavage site "ENLYFQG" encoded in DNA
    # Found on several sources too
    "Sensor_Protease": "GAA AAC CTT TAC TTC CAA GGG",

    # Linker: Flexible spacer sequence used in protein linkers and synthetic constructs
    # Found on several sources too
    "Linker": "GGGAAAAGGGAAGGGAAAGGG",

    # Downstream_Protease: Protease substrate sequence placeholder (designed experimentally for downstream protease specificity)
    # Source: Thrombin Cleave Site (https://www.sigmaaldrich.com/MA/en/technical-documents/technical-article/research-and-disease-areas/metabolism-research/thrombins?srsltid=AfmBOoqY14-8t4CqI2L9SpsX1VwsrRN1oy9Wxmn-gSw6SPub16o5pMta)
    "Downstream_Protease": "CTGGTGCCGCGCGGCAGC",

    # Logic_Module: Synthetic biology regulatory sequence with common restriction enzyme sites
    # Source: Combination of XbaI (TCTAGA), NotI (GCGGCCGC), EcoRI (GAATTC) sites frequently used in cloning
    "Logic_Module": "TCTAGAGCGGCCGCCGAATTC",

    # Output_Reporter: Partial Green Fluorescent Protein (GFP) coding sequence start
    # Source: Found on UniProt (https://www.uniprot.org/uniprotkb/P42212/entry)
    "Output_Reporter": "ATGAGCAAAGGCGAAGAACTGTTTACCGGCGTGGTGCCGATTCTGGTGGAACTGGATGGCGATGTGAACGGCCATAAATTTAGCGTGAGCGGCGAAGGCGAAGGCGATGCGACCTATGGCAAACTGACCCTGAAATTTATTTGCACCACCGGCAAACTGCCGGTGCCGTGGCCGACCCTGGTGACCACCTTTAGCTATGGCGTGCAGTGCTTTAGCCGCTATCCGGATCATATGAAACAGCATGATTTTTTTAAAAGCGCGATGCCGGAAGGCTATGTGCAGGAACGCACCATTTTTTTTAAAGATGATGGCAACTATAAAACCCGCGCGGAAGTGAAATTTGAAGGCGATACCCTGGTGAACCGCATTGAACTGAAAGGCATTGATTTTAAAGAAGATGGCAACATTCTGGGCCATAAACTGGAATATAACTATAACAGCCATAACGTGTATATTATGGCGGATAAACAGAAAAACGGCATTAAAGTGGAACTTTAAAATTCGCCATAACATTGAAGATGGGCAGCGTGCAGCTGGCGGATCATTATCAGCAGAACACCCCGATTGGCGATGGCCCGGTGCTGCTGCCGGATAACCATTATCTGAGCACCCAGAGCGCGCTGAGCAAAGATCCGAACGAAAAACGCGATCATATGGTGCTGCTGGAATTTGTGACCGCGGCGGGCATTACCCATGGCATGGATGAACTGTATAAA",

    # Stop codon: Standard stop codon used for translation termination
    # Just using TAA here since it's one of the three standard stop codons
    "Stop": "TAA",

    # PolyA signal: Canonical eukaryotic polyadenylation signal sequence
    # Commonly used sequence "AATAAA"
    "PolyA": "AATAAA",

    # ITR_3: Adeno-associated virus (AAV) 3' inverted terminal repeat, reverse complement of ITR_5
    # Source: (https://pmc.ncbi.nlm.nih.gov/articles/PMC7047122/)
    "ITR_3": "CGGCCTCGTCACGCGCCGGCAAGACGTACGGCCGCCGTAGCCG"
}


def parse_fasta(path: str):
    with open(path, 'r', encoding='utf-8') as fh:
        header, seq_chunks = None, []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header, seq_chunks = line[1:].strip(), []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks)

# Match genes to promoters ----
HEADER_RE = re.compile(r'^(?P<pid>\S+)\s*(?P<rest>.*)$')

def parse_header(header: str):
    m = HEADER_RE.match(header)
    return m.group("pid") if m else None, header

def find_promoters(promoters_fasta, genes):
    promoters = {}
    for header, seq in parse_fasta(promoters_fasta):
        pid, rest = parse_header(header)
        for g in genes:
            if re.search(r"\b" + re.escape(g) + r"\b", header, flags=re.IGNORECASE):
                promoters[g] = {"id": pid, "header": header, "sequence": seq}
    return promoters

# Just combine everything to build the circuit
def build_circuit(promoter_seq: str):
    dna_parts = [
        MODULES["ITR_5"],
        promoter_seq,
        MODULES["Kozak_Start"],
        MODULES["Sensor_Protease"],
        MODULES["Linker"],
        MODULES["Downstream_Protease"],
        MODULES["Logic_Module"],
        MODULES["Output_Reporter"],
        MODULES["Stop"],
        MODULES["PolyA"],
        MODULES["ITR_3"]
    ]
    return "".join(dna_parts)

# SBOL-like Diagram (This had to be AI generated as I couldn't find a good library for SBOL diagrams, sbol-canvas doesn't work on my system)
def draw_sbol_diagram(gene: str, promoter_info: dict, outdir: str):
    features = [
        GraphicFeature(start=0, end=100, strand=+1, color="#ffd700", label="ITR_5"),
        GraphicFeature(start=100, end=200, strand=+1, color="#32cd32", 
                       label=f"{gene}_promoter ({promoter_info.get('id','')})"),
        GraphicFeature(start=200, end=250, strand=+1, color="#1e90ff", label="Kozak"),
        GraphicFeature(start=250, end=450, strand=+1, color="#ff4500", label="Sensor_Protease"),
        GraphicFeature(start=450, end=500, strand=+1, color="#9370db", label="Linker"),
        GraphicFeature(start=500, end=700, strand=+1, color="#ff6347", label="Downstream_Protease"),
        GraphicFeature(start=700, end=900, strand=+1, color="#4682b4", label="Logic_Module"),
        GraphicFeature(start=900, end=1100, strand=+1, color="#008000", label="Output_Reporter"),
        GraphicFeature(start=1100, end=1150, strand=+1, color="#000000", label="Stop/PolyA"),
        GraphicFeature(start=1150, end=1250, strand=+1, color="#ffd700", label="ITR_3"),
    ]

    record = GraphicRecord(sequence_length=1300, features=features)
    ax, _ = record.plot(figure_width=10)

    svg_path = os.path.join(outdir, f"{gene}_circuit.svg")
    png_path = os.path.join(outdir, f"{gene}_circuit.png")

    ax.figure.savefig(svg_path)
    ax.figure.savefig(png_path, dpi=300)
    print(f"Diagram saved as {svg_path} and {png_path}")
    return svg_path, png_path


def write_dna_file(gene: str, circuit_dna: str, outdir: str):
    dna_path = os.path.join(outdir, f"{gene}_genetic_circuit.dna")
    with open(dna_path, "w", encoding="utf-8") as fh:
        fh.write(f">{gene}_genetic_circuit\n")
        # 80 lines per time to give it the DNA vibe
        for i in range(0, len(circuit_dna), 80):
            fh.write(circuit_dna[i:i+80] + "\n")
    print(f"DNA file saved as {dna_path}")
    return dna_path


def main():
    os.makedirs(CONFIG["OUTDIR"], exist_ok=True)
    genes = [line.strip() for line in open(CONFIG["GENES_FILE"], encoding="utf-8") if line.strip()]
    promoters = find_promoters(CONFIG["PROMOTERS_FASTA"], genes)

    out_csv = os.path.join(CONFIG["OUTDIR"], "promoter_circuits.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gene", "promoter_id", "header", "promoter_seq", "circuit_dna"])
        for g in genes:
            if g in promoters:
                p = promoters[g]
                circuit_dna = build_circuit(p["sequence"])
                writer.writerow([g, p["id"], p["header"], p["sequence"], circuit_dna])
                # Save diagram
                draw_sbol_diagram(g, p, CONFIG["OUTDIR"])
                # Save DNA file
                write_dna_file(g, circuit_dna, CONFIG["OUTDIR"])
            else:
                writer.writerow([g, "", "", "", ""])

    print(f"CSV summary written to {out_csv}")
    return out_csv

if __name__ == "__main__":
    main()
