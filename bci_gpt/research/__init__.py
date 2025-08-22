"""Advanced research framework for BCI-GPT academic validation and publication."""

from .experimental_framework import (
    ExperimentConfig,
    ExperimentResult,
    StatisticalValidator,
    BCIMetricsCalculator,
    ExperimentFramework
)

from .reproducibility import (
    ReproducibilityConfig,
    ExperimentState,
    ReproducibilityTracker
)

from .publication_generator import (
    PublicationConfig,
    PaperSection,
    FigureSpec,
    PublicationFigureGenerator,
    AcademicPaperGenerator
)

from .benchmark_suite import (
    BenchmarkConfig,
    BenchmarkResult,
    BaseDataset,
    BaseModel,
    SyntheticEEGDataset,
    SimpleTransformerModel,
    RandomForestBaseline,
    BenchmarkSuite
)

from .distributed_computation import (
    DistributedConfig,
    DistributedTrainer,
    RayDistributedFramework,
    DaskDistributedFramework,
    HyperparameterOptimization,
    DistributedExperimentManager
)

from .federated_learning import (
    FederatedConfig,
    ClientData,
    FederatedClient,
    FederatedAggregator,
    FederatedLearningServer
)

from .auto_ml_pipeline import (
    AutoMLConfig,
    NeuralArchitectureSpec,
    DynamicNeuralNetwork,
    NeuralArchitectureSearch,
    DataPreprocessingPipeline,
    AutoMLPipeline
)

__all__ = [
    # Experimental Framework
    'ExperimentConfig',
    'ExperimentResult', 
    'StatisticalValidator',
    'BCIMetricsCalculator',
    'ExperimentFramework',
    
    # Reproducibility
    'ReproducibilityConfig',
    'ExperimentState',
    'ReproducibilityTracker',
    
    # Publication Generation
    'PublicationConfig',
    'PaperSection',
    'FigureSpec',
    'PublicationFigureGenerator',
    'AcademicPaperGenerator',
    
    # Benchmark Suite
    'BenchmarkConfig',
    'BenchmarkResult',
    'BaseDataset',
    'BaseModel',
    'SyntheticEEGDataset',
    'SimpleTransformerModel',
    'RandomForestBaseline',
    'BenchmarkSuite',
    
    # Distributed Computation
    'DistributedConfig',
    'DistributedTrainer',
    'RayDistributedFramework',
    'DaskDistributedFramework',
    'HyperparameterOptimization',
    'DistributedExperimentManager',
    
    # Federated Learning
    'FederatedConfig',
    'ClientData',
    'FederatedClient',
    'FederatedAggregator',
    'FederatedLearningServer',
    
    # AutoML Pipeline
    'AutoMLConfig',
    'NeuralArchitectureSpec',
    'DynamicNeuralNetwork',
    'NeuralArchitectureSearch',
    'DataPreprocessingPipeline',
    'AutoMLPipeline'
]