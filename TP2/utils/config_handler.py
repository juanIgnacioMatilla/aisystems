from TP2.src.hyperparams.selection.combined_selection import CombinedSelection
from TP2.utils.hyperparams_mapping import TERMINATION_MAP, REPLACEMENT_MAP, MUTATION_MAP, CROSSOVER_MAP, SELECTION_MAP


def get_strategies(run_config):
    # Initialize Selection Strategy
    selection_params = run_config['hyperparams']['selection']
    k = selection_params['k']  # Extract k once, use it for both method_a and method_b

    # Check if the selection strategy is "combined"
    if selection_params['name'] == 'combined':
        # Load method A
        method_a_params = selection_params.pop('method_a')
        method_a_class = SELECTION_MAP[method_a_params.pop('name')]
        method_a_strategy = method_a_class(k=k, **method_a_params)

        # Load method B
        method_b_params = selection_params.pop('method_b')
        method_b_class = SELECTION_MAP[method_b_params.pop('name')]
        method_b_strategy = method_b_class(k=k, **method_b_params)

        # Load percentage_a
        percentage_a = selection_params.pop('percentage_a')

        # Create the CombinedSelection strategy
        selection_strategy = CombinedSelection(
            k=k,
            method_a=method_a_strategy,
            method_b=method_b_strategy,
            percentage_a=percentage_a
        )
    else:
        # For non-combined selection strategies
        selection_class = SELECTION_MAP[selection_params.pop('name')]
        selection_strategy = selection_class(k=k, **selection_params)

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
