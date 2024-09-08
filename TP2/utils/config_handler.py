from TP2.src.hyperparams.selection.combined_selection import CombinedSelection
from TP2.utils.hyperparams_mapping import TERMINATION_MAP, REPLACEMENT_MAP, MUTATION_MAP, CROSSOVER_MAP, SELECTION_MAP


def initialize_selection_strategy(selection_params):
    """
    Initialize the selection strategy based on the provided parameters.
    """
    k = selection_params['k']  # Extract k once for use in both method_a and method_b if needed

    if selection_params['name'] == 'combined':
        return initialize_combined_selection_strategy(selection_params, k)
    else:
        return initialize_single_selection_strategy(selection_params, k)


def initialize_combined_selection_strategy(selection_params, k):
    """
    Initialize the CombinedSelection strategy.
    """
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

    return CombinedSelection(
        k=k,
        method_a=method_a_strategy,
        method_b=method_b_strategy,
        percentage_a=percentage_a
    )


def initialize_single_selection_strategy(selection_params, k):
    """
    Initialize a single selection strategy (non-combined).
    """
    selection_name = selection_params.pop('name')
    selection_class = SELECTION_MAP[selection_name]
    return selection_class(k=k, **selection_params)


def initialize_crossover_strategy(crossover_params):
    """
    Initialize the crossover strategy based on the provided parameters.
    """
    crossover_class = CROSSOVER_MAP[crossover_params['name']]
    return crossover_class()


def initialize_mutation_strategy(mutation_params):
    """
    Initialize the mutation strategy based on the provided parameters.
    """
    mutation_class = MUTATION_MAP[mutation_params.pop('name')]
    return mutation_class(**mutation_params)


def initialize_replacement_strategy(replacement_params):
    """
    Initialize the replacement strategy based on the provided parameters and an optional selection method.
    """
    if 'selection' in replacement_params:
        replacement_selection_strategy = initialize_replacement_selection_strategy(replacement_params.pop('selection'))
    else:
        replacement_selection_strategy = None
    replacement_class = REPLACEMENT_MAP[replacement_params.pop('name')]

    if replacement_selection_strategy:
        return replacement_class(replacement_selection_strategy)
    else:
        return replacement_class()


def initialize_replacement_selection_strategy(selection_params):
    """
    Initialize the selection strategy used in replacement methods.
    """
    k = selection_params.pop('k')  # Extract k for the replacement selection

    if selection_params['name'] == 'combined':
        return initialize_combined_selection_strategy(selection_params, k)
    else:
        return initialize_single_selection_strategy(selection_params, k)


def initialize_termination_strategy(termination_params):
    """
    Initialize the termination strategy based on the provided parameters.
    """
    termination_class = TERMINATION_MAP[termination_params.pop('name')]
    return termination_class(**termination_params)


def get_strategies(run_config):
    """
    Extract and initialize all strategies from the run configuration.
    """
    # Initialize Selection Strategy
    selection_params = run_config['hyperparams']['selection']
    selection_strategy = initialize_selection_strategy(selection_params)

    # Initialize Crossover Strategy
    crossover_strategy = initialize_crossover_strategy(run_config['hyperparams']['crossover'])

    # Initialize Mutation Strategy
    mutation_strategy = initialize_mutation_strategy(run_config['hyperparams']['mutation'])

    # Initialize Replacement Strategy
    run_config['hyperparams']['replacement']['selection']['k'] = run_config['population_size']
    replacement_strategy = initialize_replacement_strategy(run_config['hyperparams']['replacement'])

    # Initialize Termination Strategy
    termination_strategy = initialize_termination_strategy(run_config['hyperparams']['termination'])

    return (
        selection_strategy,
        crossover_strategy,
        mutation_strategy,
        replacement_strategy,
        termination_strategy,
    )
