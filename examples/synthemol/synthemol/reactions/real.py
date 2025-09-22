"""SMARTS representations of the most common Enamine REAL reactions."""
from synthemol.reactions.query_mol import QueryMol
from synthemol.reactions.reaction import Reaction

REAL_REACTIONS = (
    Reaction(
        reactants=[
            QueryMol("CC(C)(C)OC(=O)[N:1]([*:2])[*:3].[*:4][N:5]([H])[*:6]"),
            QueryMol("[OH1][C:7]([*:8])=[O:9]"),
            QueryMol("[OH1][C:10]([*:11])=[O:12]"),
        ],
        product=QueryMol(
            "[*:4][N:5]([*:6])[C:7](=[O:9])[*:8].[*:3][N:1]([*:2])[C:10](=[O:12])[*:11]"
        ),
        reaction_id=275592,
    ),
    Reaction(
        reactants=[
            QueryMol("[*:1][N:2]([H])[*:3]"),
            QueryMol("[OH1][C:4]([*:5])=[O:6]"),
        ],
        product=QueryMol("[*:5][C:4](=[O:6])[N:2]([*:1])[*:3]"),
        reaction_id=22,
    ),
    Reaction(
        reactants=[
            QueryMol("[*:1][N:2]([H])[*:3]"),
            QueryMol("[OH1][C:4]([*:5])=[O:6]"),
        ],
        product=QueryMol("[*:5][C:4](=[O:6])[N:2]([*:1])[*:3]"),
        reaction_id=11,
    ),
    Reaction(
        reactants=[
            QueryMol("[*:1][N:2]([H])[*:3]"),
            QueryMol("[OH1][C:4]([*:5])=[O:6]"),
        ],
        product=QueryMol("[*:5][C:4](=[O:6])[N:2]([*:1])[*:3]"),
        reaction_id=527,
    ),
    Reaction(
        reactants=[
            QueryMol("[*:1][N:2]([H])[*:3]"),
            QueryMol("[OH1][C:4]([*:5])=[O:6]"),
        ],
        product=QueryMol("[*:5][C:4](=[O:6])[N:2]([*:1])[*:3]"),
        reaction_id=240690,
    ),
    Reaction(
        reactants=[QueryMol("[*:1][N:2]([H])[H:3]"), QueryMol("[*:4][N:5]([H])[*:6]")],
        product=QueryMol("O=C([N:2]([*:1])[H:3])[N:5]([*:4])[*:6]"),
        reaction_id=2430,
    ),
    Reaction(
        reactants=[QueryMol("[*:1][N:2]([H])[H:3]"), QueryMol("[*:4][N:5]([H])[H:6]")],
        product=QueryMol("O=C([N:2]([*:1])[H:3])[N:5]([*:4])[H:6]"),
        reaction_id=2708,
    ),
    Reaction(
        reactants=[QueryMol("[*:1][N:2]([H])[*:3]"), QueryMol("[F,Cl,Br,I][*:4]")],
        product=QueryMol("[*:1][N:2]([*:3])[*:4]"),
        reaction_id=2230,
    ),
    Reaction(
        reactants=[QueryMol("[*:1][N:2]([H])[H:3]"), QueryMol("[*:4][N:5]([H])[H:6]")],
        product=QueryMol("O=C(C(=O)[N:2]([*:1])[H:3])[N:5]([*:4])[H:6]"),
        reaction_id=2718,
    ),
    Reaction(
        reactants=[
            QueryMol("[*:1][N:2]([H])[*:3]"),
            QueryMol("[O:4]=[S:5](=[O:6])([F,Cl,Br,I])[*:7]"),
        ],
        product=QueryMol("[O:4]=[S:5](=[O:6])([*:7])[N:2]([*:1])[*:3]"),
        reaction_id=40,
    ),
    Reaction(
        reactants=[QueryMol("[*:1][N:2]([H])[*:3]"), QueryMol("[F,Cl,Br,I][*:4]")],
        product=QueryMol("[*:1][N:2]([*:3])[*:4]"),
        reaction_id=27,
    ),
    Reaction(
        reactants=[QueryMol("[*:1][N:2]([H])[*:3]"), QueryMol("[*:4][N:5]([H])[H:6]")],
        product=QueryMol("O=C(C(=O)[N:2]([*:1])[*:3])[N:5]([*:4])[H:6]"),
        reaction_id=271948,
    ),
    Reaction(
        reactants=[QueryMol("[OH1:1][C:2]([*:3])=[O:4]"), QueryMol("[F,Cl,Br,I][*:5]")],
        product=QueryMol("[O:4]=[C:2]([*:3])[O:1][*:5]"),
        reaction_id=1458,
    ),
)
