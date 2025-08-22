"""Academic publication generator for BCI-GPT research with automated LaTeX and figure generation."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd
from datetime import datetime
import logging
import re

# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PublicationConfig:
    """Configuration for academic publication generation."""
    venue: str = "NeurIPS"  # NeurIPS, ICML, Nature, IEEE_TBME, etc.
    template_style: str = "neurips"  # neurips, icml, nature, ieee
    include_appendix: bool = True
    generate_figures: bool = True
    figure_format: str = "pdf"  # pdf, png, svg
    font_family: str = "Times"
    font_size: int = 10
    line_width: float = 1.5
    figure_dpi: int = 300
    color_blind_safe: bool = True
    
    def get_style_config(self) -> Dict[str, Any]:
        """Get venue-specific style configuration."""
        styles = {
            "neurips": {
                "columns": 2,
                "page_limit": 8,
                "font": "times",
                "bibliography_style": "neurips"
            },
            "nature": {
                "columns": 1,
                "page_limit": None,
                "font": "times",
                "bibliography_style": "nature"
            },
            "ieee": {
                "columns": 2,
                "page_limit": 12,
                "font": "times",
                "bibliography_style": "IEEEtran"
            }
        }
        return styles.get(self.template_style, styles["neurips"])


@dataclass
class PaperSection:
    """Represents a section of the academic paper."""
    title: str
    content: str
    subsections: List['PaperSection'] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)


@dataclass
class FigureSpec:
    """Specification for generating publication figures."""
    figure_id: str
    title: str
    caption: str
    figure_type: str  # line_plot, bar_chart, heatmap, confusion_matrix, etc.
    data: Dict[str, Any]
    layout: str = "single"  # single, subplot, grid
    size: Tuple[float, float] = (6, 4)  # width, height in inches


class PublicationFigureGenerator:
    """Generate publication-quality figures for BCI-GPT research."""
    
    def __init__(self, config: PublicationConfig):
        self.config = config
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """Configure matplotlib for publication-quality output."""
        plt.rcParams.update({
            'font.family': self.config.font_family,
            'font.size': self.config.font_size,
            'axes.linewidth': self.config.line_width,
            'lines.linewidth': self.config.line_width,
            'patch.linewidth': self.config.line_width,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.dpi': self.config.figure_dpi,
            'savefig.dpi': self.config.figure_dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        
        if self.config.color_blind_safe:
            # Use colorblind-friendly palette
            sns.set_palette("colorblind")
    
    def generate_performance_comparison(self, 
                                      data: Dict[str, List[float]], 
                                      figure_spec: FigureSpec,
                                      output_path: Path) -> Path:
        """Generate performance comparison figure."""
        
        fig, ax = plt.subplots(1, 1, figsize=figure_spec.size)
        
        methods = list(data.keys())
        performances = [data[method] for method in methods]
        
        # Create box plot with individual points
        bp = ax.boxplot(performances, labels=methods, patch_artist=True)
        
        # Color boxes
        colors = sns.color_palette("husl", len(methods))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual data points
        for i, perf in enumerate(performances, 1):
            x = np.random.normal(i, 0.04, size=len(perf))
            ax.scatter(x, perf, alpha=0.6, s=20)
        
        # Add mean values as text
        for i, perf in enumerate(performances, 1):
            mean_val = np.mean(perf)
            ax.text(i, mean_val + 0.02, f'{mean_val:.3f}', 
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Accuracy')
        ax.set_title(figure_spec.title)
        ax.grid(True, alpha=0.3)
        
        # Save figure
        figure_path = output_path / f"{figure_spec.figure_id}.{self.config.figure_format}"
        plt.savefig(figure_path, format=self.config.figure_format)
        plt.close()
        
        logger.info(f"Generated performance comparison: {figure_path}")
        return figure_path
    
    def generate_ablation_study(self, 
                              data: Dict[str, float], 
                              figure_spec: FigureSpec,
                              output_path: Path) -> Path:
        """Generate ablation study figure."""
        
        fig, ax = plt.subplots(1, 1, figsize=figure_spec.size)
        
        components = list(data.keys())
        performance_drops = list(data.values())
        
        # Sort by performance drop (importance)
        sorted_pairs = sorted(zip(components, performance_drops), key=lambda x: x[1], reverse=True)
        components, performance_drops = zip(*sorted_pairs)
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(components)), performance_drops, alpha=0.8)
        
        # Color bars by importance
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.Reds(0.3 + 0.7 * i / len(bars)))
        
        ax.set_yticks(range(len(components)))
        ax.set_yticklabels(components)
        ax.set_xlabel('Performance Drop (%)')
        ax.set_title(figure_spec.title)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(performance_drops):
            ax.text(v + 0.1, i, f'{v:.1f}%', va='center')
        
        plt.tight_layout()
        
        figure_path = output_path / f"{figure_spec.figure_id}.{self.config.figure_format}"
        plt.savefig(figure_path, format=self.config.figure_format)
        plt.close()
        
        logger.info(f"Generated ablation study: {figure_path}")
        return figure_path
    
    def generate_attention_heatmap(self, 
                                 attention_weights: np.ndarray,
                                 eeg_channels: List[str],
                                 tokens: List[str],
                                 figure_spec: FigureSpec,
                                 output_path: Path) -> Path:
        """Generate attention visualization heatmap."""
        
        fig, ax = plt.subplots(1, 1, figsize=figure_spec.size)
        
        # Create heatmap
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticks(range(len(eeg_channels)))
        ax.set_yticklabels(eeg_channels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        
        ax.set_xlabel('Tokens')
        ax.set_ylabel('EEG Channels')
        ax.set_title(figure_spec.title)
        
        plt.tight_layout()
        
        figure_path = output_path / f"{figure_spec.figure_id}.{self.config.figure_format}"
        plt.savefig(figure_path, format=self.config.figure_format)
        plt.close()
        
        logger.info(f"Generated attention heatmap: {figure_path}")
        return figure_path
    
    def generate_temporal_analysis(self, 
                                 time_series_data: Dict[str, np.ndarray],
                                 time_points: np.ndarray,
                                 figure_spec: FigureSpec,
                                 output_path: Path) -> Path:
        """Generate temporal analysis figure."""
        
        fig, ax = plt.subplots(1, 1, figsize=figure_spec.size)
        
        for label, data in time_series_data.items():
            # Calculate mean and standard error
            mean_data = np.mean(data, axis=0)
            se_data = np.std(data, axis=0) / np.sqrt(data.shape[0])
            
            # Plot line with error band
            ax.plot(time_points, mean_data, label=label, linewidth=2)
            ax.fill_between(time_points, 
                           mean_data - se_data, 
                           mean_data + se_data, 
                           alpha=0.3)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(figure_spec.title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        figure_path = output_path / f"{figure_spec.figure_id}.{self.config.figure_format}"
        plt.savefig(figure_path, format=self.config.figure_format)
        plt.close()
        
        logger.info(f"Generated temporal analysis: {figure_path}")
        return figure_path
    
    def generate_confusion_matrix(self, 
                                cm: np.ndarray,
                                class_labels: List[str],
                                figure_spec: FigureSpec,
                                output_path: Path) -> Path:
        """Generate confusion matrix visualization."""
        
        fig, ax = plt.subplots(1, 1, figsize=figure_spec.size)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black")
        
        ax.set_xticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticks(range(len(class_labels)))
        ax.set_yticklabels(class_labels)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(figure_spec.title)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        figure_path = output_path / f"{figure_spec.figure_id}.{self.config.figure_format}"
        plt.savefig(figure_path, format=self.config.figure_format)
        plt.close()
        
        logger.info(f"Generated confusion matrix: {figure_path}")
        return figure_path


class AcademicPaperGenerator:
    """Generate complete academic papers for BCI-GPT research."""
    
    def __init__(self, config: PublicationConfig):
        self.config = config
        self.figure_generator = PublicationFigureGenerator(config)
        
    def generate_paper(self, 
                      paper_data: Dict[str, Any],
                      output_path: Path) -> Dict[str, Path]:
        """Generate complete academic paper with figures."""
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate figures first
        figure_paths = self._generate_all_figures(paper_data, output_path)
        
        # Generate LaTeX paper
        latex_path = self._generate_latex_paper(paper_data, figure_paths, output_path)
        
        # Generate bibliography
        bib_path = self._generate_bibliography(paper_data, output_path)
        
        # Generate supplementary materials
        supp_path = self._generate_supplementary(paper_data, output_path)
        
        return {
            'latex': latex_path,
            'bibliography': bib_path,
            'supplementary': supp_path,
            'figures': figure_paths
        }
    
    def _generate_all_figures(self, 
                            paper_data: Dict[str, Any], 
                            output_path: Path) -> Dict[str, Path]:
        """Generate all figures for the paper."""
        
        figure_paths = {}
        figures_dir = output_path / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        if 'figures' not in paper_data:
            return figure_paths
        
        for figure_spec_dict in paper_data['figures']:
            figure_spec = FigureSpec(**figure_spec_dict)
            
            if figure_spec.figure_type == 'performance_comparison':
                path = self.figure_generator.generate_performance_comparison(
                    figure_spec.data, figure_spec, figures_dir
                )
            elif figure_spec.figure_type == 'ablation_study':
                path = self.figure_generator.generate_ablation_study(
                    figure_spec.data, figure_spec, figures_dir
                )
            elif figure_spec.figure_type == 'attention_heatmap':
                path = self.figure_generator.generate_attention_heatmap(
                    figure_spec.data['attention_weights'],
                    figure_spec.data['eeg_channels'],
                    figure_spec.data['tokens'],
                    figure_spec, figures_dir
                )
            elif figure_spec.figure_type == 'temporal_analysis':
                path = self.figure_generator.generate_temporal_analysis(
                    figure_spec.data['time_series'],
                    figure_spec.data['time_points'],
                    figure_spec, figures_dir
                )
            elif figure_spec.figure_type == 'confusion_matrix':
                path = self.figure_generator.generate_confusion_matrix(
                    figure_spec.data['confusion_matrix'],
                    figure_spec.data['class_labels'],
                    figure_spec, figures_dir
                )
            else:
                logger.warning(f"Unknown figure type: {figure_spec.figure_type}")
                continue
            
            figure_paths[figure_spec.figure_id] = path
        
        return figure_paths
    
    def _generate_latex_paper(self, 
                            paper_data: Dict[str, Any], 
                            figure_paths: Dict[str, Path],
                            output_path: Path) -> Path:
        """Generate LaTeX paper."""
        
        style_config = self.config.get_style_config()
        
        # Paper header
        latex_content = self._generate_latex_header(paper_data, style_config)
        
        # Abstract
        latex_content += self._generate_abstract(paper_data)
        
        # Introduction
        latex_content += self._generate_introduction(paper_data)
        
        # Related Work
        latex_content += self._generate_related_work(paper_data)
        
        # Methodology
        latex_content += self._generate_methodology(paper_data)
        
        # Experiments
        latex_content += self._generate_experiments(paper_data, figure_paths)
        
        # Results
        latex_content += self._generate_results(paper_data, figure_paths)
        
        # Discussion
        latex_content += self._generate_discussion(paper_data)
        
        # Conclusion
        latex_content += self._generate_conclusion(paper_data)
        
        # Bibliography
        latex_content += "\\bibliographystyle{" + style_config['bibliography_style'] + "}\n"
        latex_content += "\\bibliography{references}\n\n"
        
        # Appendix (if requested)
        if self.config.include_appendix:
            latex_content += self._generate_appendix(paper_data)
        
        latex_content += "\\end{document}\n"
        
        # Save LaTeX file
        latex_path = output_path / "paper.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"Generated LaTeX paper: {latex_path}")
        return latex_path
    
    def _generate_latex_header(self, 
                             paper_data: Dict[str, Any], 
                             style_config: Dict[str, Any]) -> str:
        """Generate LaTeX document header."""
        
        header = f"""\\documentclass[{style_config['font']},10pt,twocolumn]{{article}}

% Packages
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{algorithm}}
\\usepackage{{algorithmic}}
\\usepackage{{hyperref}}
\\usepackage{{natbib}}
\\usepackage{{url}}
\\usepackage{{subcaption}}

% Title and authors
\\title{{{paper_data.get('title', 'BCI-GPT Research Paper')}}}

\\author{{
"""
        
        # Add authors
        authors = paper_data.get('authors', [])
        for i, author in enumerate(authors):
            if i > 0:
                header += " \\and "
            header += f"{author['name']}"
            if 'affiliation' in author:
                header += f"\\\\{author['affiliation']}"
            if 'email' in author:
                header += f"\\\\\\texttt{{{author['email']}}}"
        
        header += "}\n\n\\begin{document}\n\\maketitle\n\n"
        
        return header
    
    def _generate_abstract(self, paper_data: Dict[str, Any]) -> str:
        """Generate abstract section."""
        
        abstract = paper_data.get('abstract', 'Abstract content here.')
        
        return f"""\\begin{{abstract}}
{abstract}
\\end{{abstract}}

"""
    
    def _generate_introduction(self, paper_data: Dict[str, Any]) -> str:
        """Generate introduction section."""
        
        intro = paper_data.get('introduction', 'Introduction content here.')
        
        return f"""\\section{{Introduction}}

{intro}

"""
    
    def _generate_related_work(self, paper_data: Dict[str, Any]) -> str:
        """Generate related work section."""
        
        related_work = paper_data.get('related_work', 'Related work content here.')
        
        return f"""\\section{{Related Work}}

{related_work}

"""
    
    def _generate_methodology(self, paper_data: Dict[str, Any]) -> str:
        """Generate methodology section."""
        
        methodology = paper_data.get('methodology', 'Methodology content here.')
        
        # Add mathematical equations if provided
        equations = paper_data.get('equations', {})
        for eq_name, eq_latex in equations.items():
            methodology += f"\n\\begin{{equation}}\n{eq_latex}\n\\label{{eq:{eq_name}}}\n\\end{{equation}}\n"
        
        return f"""\\section{{Methodology}}

{methodology}

"""
    
    def _generate_experiments(self, 
                            paper_data: Dict[str, Any], 
                            figure_paths: Dict[str, Path]) -> str:
        """Generate experiments section."""
        
        experiments = paper_data.get('experiments', 'Experimental setup here.')
        
        return f"""\\section{{Experiments}}

{experiments}

"""
    
    def _generate_results(self, 
                        paper_data: Dict[str, Any], 
                        figure_paths: Dict[str, Path]) -> str:
        """Generate results section with figures and tables."""
        
        results = paper_data.get('results', 'Results content here.')
        
        results_section = f"\\section{{Results}}\n\n{results}\n\n"
        
        # Add figures
        for figure_id, figure_path in figure_paths.items():
            figure_data = next((fig for fig in paper_data.get('figures', []) 
                              if fig['figure_id'] == figure_id), None)
            if figure_data:
                results_section += f"""\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=\\columnwidth]{{figures/{figure_path.name}}}
\\caption{{{figure_data['caption']}}}
\\label{{fig:{figure_id}}}
\\end{{figure}}

"""
        
        # Add tables if provided
        tables = paper_data.get('tables', [])
        for table in tables:
            results_section += self._generate_latex_table(table)
        
        return results_section
    
    def _generate_discussion(self, paper_data: Dict[str, Any]) -> str:
        """Generate discussion section."""
        
        discussion = paper_data.get('discussion', 'Discussion content here.')
        
        return f"""\\section{{Discussion}}

{discussion}

"""
    
    def _generate_conclusion(self, paper_data: Dict[str, Any]) -> str:
        """Generate conclusion section."""
        
        conclusion = paper_data.get('conclusion', 'Conclusion content here.')
        
        return f"""\\section{{Conclusion}}

{conclusion}

"""
    
    def _generate_appendix(self, paper_data: Dict[str, Any]) -> str:
        """Generate appendix section."""
        
        appendix = paper_data.get('appendix', '')
        
        if not appendix:
            return ""
        
        return f"""\\appendix

\\section{{Supplementary Materials}}

{appendix}

"""
    
    def _generate_latex_table(self, table_data: Dict[str, Any]) -> str:
        """Generate LaTeX table from data."""
        
        caption = table_data.get('caption', 'Table caption')
        label = table_data.get('label', 'table1')
        headers = table_data.get('headers', [])
        rows = table_data.get('rows', [])
        
        # Create column specification
        num_cols = len(headers)
        col_spec = 'l' * num_cols
        
        table_latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{tab:{label}}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
"""
        
        # Add header row
        table_latex += " & ".join(headers) + " \\\\\n\\midrule\n"
        
        # Add data rows
        for row in rows:
            table_latex += " & ".join(str(cell) for cell in row) + " \\\\\n"
        
        table_latex += """\\bottomrule
\\end{tabular}
\\end{table}

"""
        
        return table_latex
    
    def _generate_bibliography(self, 
                             paper_data: Dict[str, Any], 
                             output_path: Path) -> Path:
        """Generate bibliography file."""
        
        references = paper_data.get('references', [])
        
        bib_content = "% BCI-GPT Research Bibliography\n\n"
        
        for ref in references:
            if ref['type'] == 'article':
                bib_content += f"""@article{{{ref['key']},
  title={{{ref['title']}}},
  author={{{ref['author']}}},
  journal={{{ref['journal']}}},
  volume={{{ref.get('volume', '')}}},
  number={{{ref.get('number', '')}}},
  pages={{{ref.get('pages', '')}}},
  year={{{ref['year']}}},
  publisher={{{ref.get('publisher', '')}}}
}}

"""
            elif ref['type'] == 'inproceedings':
                bib_content += f"""@inproceedings{{{ref['key']},
  title={{{ref['title']}}},
  author={{{ref['author']}}},
  booktitle={{{ref['booktitle']}}},
  pages={{{ref.get('pages', '')}}},
  year={{{ref['year']}}},
  organization={{{ref.get('organization', '')}}}
}}

"""
        
        bib_path = output_path / "references.bib"
        with open(bib_path, 'w') as f:
            f.write(bib_content)
        
        logger.info(f"Generated bibliography: {bib_path}")
        return bib_path
    
    def _generate_supplementary(self, 
                              paper_data: Dict[str, Any], 
                              output_path: Path) -> Path:
        """Generate supplementary materials."""
        
        supp_content = paper_data.get('supplementary', '')
        
        if not supp_content:
            supp_content = "Supplementary materials will be provided upon request."
        
        supp_path = output_path / "supplementary.tex"
        with open(supp_path, 'w') as f:
            f.write(f"""\\documentclass{{article}}
\\usepackage{{amsmath,amssymb,graphicx,booktabs}}

\\title{{Supplementary Materials: {paper_data.get('title', 'BCI-GPT Research')}}}

\\begin{{document}}
\\maketitle

{supp_content}

\\end{{document}}
""")
        
        logger.info(f"Generated supplementary materials: {supp_path}")
        return supp_path


# Example usage for BCI-GPT research
if __name__ == "__main__":
    # Configuration for NeurIPS submission
    config = PublicationConfig(
        venue="NeurIPS",
        template_style="neurips",
        include_appendix=True,
        generate_figures=True,
        figure_format="pdf"
    )
    
    # Sample paper data
    paper_data = {
        'title': 'Cross-Modal Attention Mechanisms for Real-Time Brain-Computer Interface Communication',
        'authors': [
            {'name': 'Daniel Schmidt', 'affiliation': 'Terragon Labs', 'email': 'daniel@terragonlabs.com'},
            {'name': 'Research Team', 'affiliation': 'University Research Center'}
        ],
        'abstract': 'We present a novel approach for brain-computer interface communication using cross-modal attention mechanisms between EEG signals and large language models...',
        'introduction': 'Brain-computer interfaces (BCIs) have emerged as a promising technology for direct neural communication...',
        'methodology': 'Our approach combines transformer-based EEG encoders with cross-attention fusion layers...',
        'results': 'Experimental results demonstrate significant improvements over baseline methods...',
        'figures': [
            {
                'figure_id': 'performance_comparison',
                'title': 'Performance Comparison Across Methods',
                'caption': 'Comparison of classification accuracy across different BCI decoding methods. Our approach (BCI-GPT) significantly outperforms baseline methods.',
                'figure_type': 'performance_comparison',
                'data': {
                    'Baseline CNN': np.random.normal(0.75, 0.05, 30),
                    'LSTM Decoder': np.random.normal(0.80, 0.04, 30),
                    'Transformer': np.random.normal(0.82, 0.04, 30),
                    'BCI-GPT (Ours)': np.random.normal(0.87, 0.03, 30)
                }
            },
            {
                'figure_id': 'ablation_study',
                'title': 'Ablation Study Results',
                'caption': 'Impact of different components on overall performance. Cross-modal attention shows the highest importance.',
                'figure_type': 'ablation_study',
                'data': {
                    'Cross-Modal Attention': 8.5,
                    'Temporal Convolution': 5.2,
                    'Spatial Attention': 3.8,
                    'Layer Normalization': 2.1
                }
            }
        ],
        'references': [
            {
                'type': 'article',
                'key': 'sutskever2014sequence',
                'title': 'Sequence to sequence learning with neural networks',
                'author': 'Sutskever, Ilya and Vinyals, Oriol and Le, Quoc V',
                'journal': 'Advances in neural information processing systems',
                'year': '2014'
            }
        ]
    }
    
    # Generate paper
    generator = AcademicPaperGenerator(config)
    output_path = Path("./generated_paper")
    
    paper_files = generator.generate_paper(paper_data, output_path)
    
    print("Generated academic paper:")
    for file_type, path in paper_files.items():
        print(f"  {file_type}: {path}")
    
    print(f"\nPaper files saved to: {output_path}")
    print("To compile LaTeX: cd generated_paper && pdflatex paper.tex")