import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def check_how_many_copies_of_genes(person, one_gene, two_genes):
    """util to check if person has any of the problematioc genes

    Args:
        person (_type_): _description_
        one_gene (_type_): _description_
        two_genes (_type_): _description_

    Returns:
        int: 0, 1 or 2
    """
    if person in one_gene:
        return 1
    if person in two_genes:
        return 2
    else:
        return 0


def probs_if_no_parent(copies_gene, has_trait):
    """returns the maths for working out if person has the trait with
    not parent

    Args:
        copies_gene (_type_): _description_
        has_trait (bool): _description_

    Returns:
        float: probability they have the trait
    """
    prob_genes = PROBS['gene'][copies_gene]
    probs_trait = PROBS['trait'][copies_gene][has_trait]
    return prob_genes * probs_trait


def probs_if_known_parent(person, people, one_gene, two_genes, have_trait):
    mother = people[person]["mother"]
    genes_mother = check_how_many_copies_of_genes(mother, one_gene, two_genes)
    father = people[person]["father"]
    genes_father = check_how_many_copies_of_genes(father, one_gene, two_genes)
    # with genes each parent can only send 1 copy,
    # two copies, 1 gene
    # one copy, 0 or 1 gene
    # no copies, 0 genes
    prob_mother_pass_gene = genes_mother / 2
    prob_father_pass_gene = genes_father / 2
    # probs pass genes and don't mutate
    prob_mother_pass_no_mutate = prob_mother_pass_gene * (1 - PROBS["mutation"])
    prob_father_pass_no_mutate = prob_father_pass_gene * (1 - PROBS["mutation"])
    # prob pass genes and do mutate
    prob_father_pass_do_mutate = prob_father_pass_gene * PROBS["mutation"]
    prob_mother_pass_do_mutate = prob_mother_pass_gene * PROBS["mutation"]
    # prob dont pass gene and don't mutate
    prob_father_no_pass_no_mutate = prob_father_pass_gene * (1 - PROBS["mutation"])
    prob_mother_no_pass_no_mutate = prob_mother_pass_gene * (1 - PROBS["mutation"])
    # prob don't pass genes and do mutate
    prob_father_no_pass_do_mutate = prob_father_pass_gene * PROBS["mutation"]
    prob_mother_no_pass_do_mutate = prob_mother_pass_gene * PROBS["mutation"]
    # child genes
    genes_child = check_how_many_copies_of_genes(person, one_gene, two_genes)
    if genes_child == 0:
        # a - father pass and mutate, mother not pass
        # b - no one pass
        # c - father not pass and mother pass and mutate
        # d - both pass and both mutate
        a = prob_father_pass_do_mutate * prob_mother_no_pass_no_mutate
        b = prob_father_no_pass_no_mutate * prob_mother_no_pass_no_mutate
        c = prob_father_no_pass_no_mutate * prob_mother_pass_do_mutate
        d = prob_father_pass_do_mutate * prob_mother_pass_do_mutate
        probability = a + b + c + d
    elif genes_child == 1:
        # a - father pass, no mutate, mother dont, no mutate
        # b - father pass, no mutate, mother pass, mutate
        # c - father pass, mutate, mother dont pass, mutate
        # d - father pass, mutate, mother pass, no mutate
        # e - father no pass, no mutate, mother no pass, mutate
        # f - father no pass, no mutate, mother pass, no mutate
        # g - father no pass, mutate, mother no pass, no mutate
        # h - father no pass, mutate, mother pass, mutate
        a = prob_father_pass_no_mutate * prob_mother_no_pass_no_mutate
        b = prob_father_pass_no_mutate * prob_mother_pass_do_mutate
        c = prob_father_pass_do_mutate * prob_mother_no_pass_do_mutate
        d = prob_father_pass_do_mutate * prob_mother_pass_no_mutate
        e = prob_father_no_pass_no_mutate * prob_mother_no_pass_no_mutate
        f = prob_father_no_pass_no_mutate * prob_mother_pass_no_mutate
        g = prob_father_no_pass_do_mutate * prob_mother_no_pass_no_mutate
        h = prob_father_no_pass_do_mutate * prob_mother_pass_do_mutate
        probability = a + b + c + d + e + f + g + h
    elif genes_child == 2:
        # a - father pass, no mutate, mother pass, no mutate
        # b - father pass, no mutate, mother no pass, mutate
        # c - father no pass, mutate, mother pass, no mutate
        # d - father no pass, mutate, mother no pass, mutate
        a = prob_father_pass_no_mutate * prob_mother_pass_no_mutate
        b = prob_father_pass_no_mutate * prob_mother_no_pass_do_mutate
        c = prob_father_no_pass_do_mutate * prob_mother_pass_no_mutate
        d = prob_father_no_pass_do_mutate * prob_mother_no_pass_do_mutate
        probability = a + b + c + d
    return probability

def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # start at 1 probability and multiply to get lower values
    probability = 1

    for person in people:
        gene_copies = check_how_many_copies_of_genes(person, one_gene, two_genes)
        if person in have_trait:
            has_trait = True
        else:
            has_trait = False
        # check if no parent and work out the probability
        if people[person]["mother"] == None:
            probability *= probs_if_no_parent(gene_copies, has_trait)
        # otherwise
        else:
            probability *= (probs_if_known_parent(person, people, one_gene, two_genes, have_trait) * PROBS['trait'][gene_copies][has_trait])

    return probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    raise NotImplementedError


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
