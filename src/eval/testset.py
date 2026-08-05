EVAL_QUESTIONS = [
    
    {
        "question": "What is tf.GradientTape used for?",
        "expected_source": "autodiff.ipynb",
        "expected_answer_summary": "tf.GradientTape records operations for automatic differentiation, allowing TensorFlow to compute gradients of a computation with respect to its input variables.",
    },
    {
        "question": "What is a Keras Sequential model?",
        "expected_source": "keras.md",
        "expected_answer_summary": "The Sequential model is the simplest type of Keras model: a linear stack of layers. For more complex architectures you can use the functional API or subclass tf.keras.Model.",
    },
    {
        "question": "How do you save a trained model using SavedModel?",
        "expected_source": "saved_model.ipynb",
        "expected_answer_summary": "Use tf.saved_model.save(model, path) to save a model, and tf.saved_model.load(path) to load it back.",
    },
    {
        "question": "What is the difference between a tf.Tensor and a tf.Variable?",
        "expected_source": "variable.ipynb",
        "expected_answer_summary": "A tf.Variable represents a tensor whose value can be changed by running ops on it, used to store shared, persistent state such as model weights, whereas a plain tf.Tensor's value is fixed once created.",
    },
    {
        "question": "How do you create a checkpoint to save model weights during training?",
        "expected_source": "checkpoint.ipynb",
        "expected_answer_summary": "Use tf.train.Checkpoint to track objects like models and optimizers, and CheckpointManager to manage multiple saved checkpoints.",
    },
    {
        "question": "What does tf.function do?",
        "expected_source": "function.ipynb",
        "expected_answer_summary": "tf.function compiles a Python function into a callable TensorFlow graph, which can improve performance by tracing the function into a graph representation.",
    },
    {
        "question": "What is a RaggedTensor?",
        "expected_source": "ragged_tensor.ipynb",
        "expected_answer_summary": "A RaggedTensor is a tensor with one or more ragged (variable-length) dimensions, used to represent data like nested lists of varying lengths.",
    },

    {
        "question": "What's the difference between the Keras functional API and subclassing tf.keras.Model?",
        "expected_source": "keras.md",
        "expected_answer_summary": "The functional API builds models by specifying inputs and outputs as a graph of layers, while subclassing tf.keras.Model gives full flexibility to define custom forward passes in Python code, at the cost of some built-in features like model reconstruction.",
    },
    {
        "question": "How can you control which variables tf.GradientTape watches?",
        "expected_source": "advanced_autodiff.ipynb",
        "expected_answer_summary": "By default the tape watches trainable variables automatically, but you can call tape.watch() to track additional tensors, or use watch_accessed_variables=False for manual control.",
    },
    {
        "question": "How do you distribute training across multiple GPUs in TensorFlow?",
        "expected_source": "distributed_training.ipynb",
        "expected_answer_summary": "TensorFlow provides tf.distribute.Strategy APIs, such as MirroredStrategy, to distribute training across multiple GPUs with minimal code changes.",
    },
    {
        "question": "What is mixed precision training and why would you use it?",
        "expected_source": "mixed_precision.ipynb",
        "expected_answer_summary": "Mixed precision training uses both 16-bit and 32-bit floating point types during training to speed up computation and reduce memory usage while maintaining model accuracy.",
    },
    {
        "question": "How do you efficiently load and preprocess data for training?",
        "expected_source": "data.ipynb",
        "expected_answer_summary": "The tf.data API lets you build input pipelines from data sources, applying transformations like batching, shuffling, and prefetching for efficient training.",
    },
    {
        "question": "What is the difference between a SparseTensor and a regular dense tensor?",
        "expected_source": "sparse_tensor.ipynb",
        "expected_answer_summary": "A SparseTensor stores only non-zero values and their indices, which is more memory-efficient than a dense tensor when most values are zero.",
    },

    {
        "question": "How does migrating from a tf.estimator.Estimator to Keras affect how you write a custom training step?",
        "expected_source": "migrate/migrating_estimator.ipynb",
        "expected_answer_summary": "In TF1, Estimators use a model_fn to define training; in TF2 with Keras, you instead subclass tf.keras.Model and override train_step, or use Model.compile and Model.fit with a custom training loop via GradientTape.",
    },
    {
        "question": "How do modules, layers, and Keras models relate to each other in TensorFlow?",
        "expected_source": "intro_to_modules.ipynb",
        "expected_answer_summary": "tf.Module is the base class for managing variables; tf.keras.layers.Layer builds on Module to add layer-specific functionality, and tf.keras.Model builds on Layer to add training and saving features like fit and save.",
    },
    {
        "question": "What considerations are involved when using TPUs versus GPUs for training?",
        "expected_source": "tpu.ipynb",
        "expected_answer_summary": "TPUs require a distribution strategy (TPUStrategy) and connection setup, and work best with large batch sizes and models that fit TPU's architecture, whereas GPU training via MirroredStrategy is more straightforward for typical hardware setups.",
    },
    {
        "question": "How do custom gradients interact with SavedModel when exporting a model?",
        "expected_source": "advanced_autodiff.ipynb",
        "expected_answer_summary": "Custom gradients can be saved to a SavedModel using tf.saved_model.SaveOptions(experimental_custom_gradients=True), but the gradient function must be traceable for this to work correctly.",
    },
    {
        "question": "What is DTensor and what problem does it solve?",
        "expected_source": "dtensor_overview.ipynb",
        "expected_answer_summary": "DTensor is a TensorFlow extension for synchronous distributed computing that provides a global programming model, letting developers write applications that operate on tensors globally while DTensor manages sharding and distribution across devices internally via SPMD expansion.",
    },
    {
        "question": "What tools does the TensorFlow Profiler provide for performance analysis?",
        "expected_source": "profiler.md",
        "expected_answer_summary": "The Profiler, accessed from the Profile tab in TensorBoard, offers tools including the Overview Page, Input Pipeline Analyzer, TensorFlow Stats, Trace Viewer, GPU Kernel Stats, Memory Profile Tool, and Pod Viewer.",
    },
    {
        "question": "Who should use the TensorFlow Core low-level APIs instead of Keras?",
        "expected_source": "core/index.md",
        "expected_answer_summary": "The Core APIs are aimed at researchers building highly configurable models, developers using TensorFlow as a high-performance scientific computing platform, framework authors building tools on top of TensorFlow, and Keras users who need to extend workflows with custom components; general ML needs are better served by high-level APIs like Keras.",
    },

    {
        "question": "What is the syntax for building a convolutional neural network image classifier in the TensorFlow tutorials section?",
        "expected_source": None,
        "expected_answer_summary": "Not covered — this is tutorial content, not part of the ingested guide/ documentation, so the system should indicate it doesn't have enough information rather than guessing.",
    },
    {
        "question": "What is PyTorch's equivalent of tf.GradientTape?",
        "expected_source": None,
        "expected_answer_summary": "Not covered — the knowledge base is scoped to TensorFlow's own guide documentation and does not contain comparisons to other frameworks.",
    },
    {
        "question": "What was announced at the most recent TensorFlow Dev Summit?",
        "expected_source": None,
        "expected_answer_summary": "Not covered — this is not static documentation content and would not appear in the guide corpus.",
    },
]