from TP2.utils.hyperparams_mapping import TERMINATION_MAP, REPLACEMENT_MAP, MUTATION_MAP, CROSSOVER_MAP, SELECTION_MAP


def get_strategies(run_config):
    # Initialize Selection Strategy
    selection_params = run_config['hyperparams']['selection']
    selection_class = SELECTION_MAP[selection_params.pop('name')]
    selection_strategy = selection_class(**selection_params)

    # Initialize Crossover Strategy (no params needed)
    crossover_class = CROSSOVER_MAP[run_config['hyperparams']['crossover']['name']]
    crossover_strategy = crossover_class()

    # Initialize Mutation Strategy
    mutation_params = run_config['hyperparams']['mutation']
    mutation_class = MUTATION_MAP[mutation_params.pop('name')]
    mutation_strategy = mutation_class(**mutation_params)

    # Initialize Replacement Strategy (no params needed)
    replacement_class = REPLACEMENT_MAP[run_config['hyperparams']['replacement']['name']]
    replacement_strategy = replacement_class()

    # Initialize Termination Strategy
    termination_params = run_config['hyperparams']['termination']
    termination_class = TERMINATION_MAP[termination_params.pop('name')]
    termination_strategy = termination_class(**termination_params)
    return (
        selection_strategy,
        crossover_strategy,
        mutation_strategy,
        replacement_strategy,
        termination_strategy,
    )
