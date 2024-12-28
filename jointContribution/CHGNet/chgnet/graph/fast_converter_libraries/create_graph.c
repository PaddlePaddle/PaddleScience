#include "uthash.h"
#include <stdbool.h>

typedef struct _UndirectedEdge UndirectedEdge;
typedef struct _DirectedEdge DirectedEdge;
typedef struct _Node Node;
typedef struct _NodeIndexPair NodeIndexPair;
typedef struct _LongToDirectedEdgeList LongToDirectedEdgeList;
typedef struct _ReturnElems2 ReturnElems2;

// NOTE: This code was mainly written to replicate the original add_edges method
// in the graph class in chgnet.graph.graph such that anyone familiar with that code should be able to pick up this
// code pretty easily.

int64_t MEM_ERR = 100;

typedef struct _Node {
    int64_t index;
    LongToDirectedEdgeList* neighbors; // Assuming neighbors can only be directed edge. Key is dest_node, value is DirectedEdge struct
    int64_t num_neighbors;
} Node;

typedef struct _NodeIndexPair {
    int64_t center;
    int64_t neighbor;
} NodeIndexPair;

typedef struct _UndirectedEdge {
    NodeIndexPair nodes;
    int64_t index;
    int64_t* directed_edge_indices;
    int64_t num_directed_edges;
    double distance;
} UndirectedEdge;

typedef struct _DirectedEdge {
    NodeIndexPair nodes;
    int64_t index;
    const int64_t* image; // Only access the first 3, never edit
    int64_t undirected_edge_index;
    double distance;
} DirectedEdge;

typedef struct _StructToUndirectedEdgeList {
    NodeIndexPair key;
    UndirectedEdge** undirected_edges_list;
    int num_undirected_edges_in_group;
    UT_hash_handle hh;
} StructToUndirectedEdgeList;

typedef struct _LongToDirectedEdgeList {
    int64_t key;
    DirectedEdge** directed_edges_list;
    int num_directed_edges_in_group;
    UT_hash_handle hh;
} LongToDirectedEdgeList;


typedef struct _ReturnElems2 {
    int64_t num_nodes;
    int64_t num_directed_edges;
    int64_t num_undirected_edges;
    Node* nodes;
    UndirectedEdge** undirected_edges_list;
    DirectedEdge** directed_edges_list;
} ReturnElems2;

bool find_in_undirected(NodeIndexPair* tmp, StructToUndirectedEdgeList** undirected_edges, StructToUndirectedEdgeList** found_entry);
void directed_to_undirected(DirectedEdge* directed, UndirectedEdge* undirected, int64_t undirected_index);
void create_new_undirected_edges_entry(StructToUndirectedEdgeList** undirected_edges, NodeIndexPair* tmp, UndirectedEdge* new_undirected_edge);
void append_to_undirected_edges_tmp(UndirectedEdge* undirected, StructToUndirectedEdgeList** undirected_edges, NodeIndexPair* tmp);
void append_to_undirected_edges_list(UndirectedEdge** undirected_edges_list, UndirectedEdge* to_add, int64_t* num_undirected_edges);
void append_to_directed_edges_list(DirectedEdge** directed_edges_list, DirectedEdge* to_add, int64_t* num_directed_edges);
void add_neighbors_to_node(Node* node, int64_t neighbor_index, DirectedEdge* directed_edge);
void print_neighbors(Node* node);
void append_to_directed_edge_indices(UndirectedEdge* undirected_edge, int64_t directed_edge_index);
bool is_reversed_directed_edge(DirectedEdge* directed_edge1, DirectedEdge* directed_edge2);
void free_undirected_edges(StructToUndirectedEdgeList** undirected_edges);
void free_LongToDirectedEdgeList_in_nodes(Node* nodes, int64_t num_nodes);


Node* create_nodes(int64_t num_nodes) {
    Node* Nodes = (Node*) malloc(sizeof(Node) * num_nodes);

    if (Nodes == NULL) {
        return NULL;
    }

    for (int64_t i = 0; i < num_nodes; i++) {
        Nodes[i].index = i;
        Nodes[i].num_neighbors = 0;

        // Initialize the uthash
        Nodes[i].neighbors = NULL;
    }

    return Nodes;
}

ReturnElems2* create_graph(
        int64_t* center_indices,
        int64_t num_edges,
        int64_t* neighbor_indices,
        int64_t* images, // contiguous memory (row-major) of image elements (total of n_e * 3 integers)
        double* distances,
        int64_t num_atoms
    ) {
    // Initialize pertinent data structures ---------------------
    Node* nodes = create_nodes(num_atoms);

    DirectedEdge** directed_edges_list = calloc(num_edges, sizeof(DirectedEdge));
    int64_t num_directed_edges = 0;

    // There will never be more undirected edges than directed edges
    UndirectedEdge** undirected_edges_list = calloc(num_edges, sizeof(UndirectedEdge));
    int64_t num_undirected_edges = 0;
    StructToUndirectedEdgeList* undirected_edges = NULL;

    // Pointer to beginning of list of UndirectedEdges corresponding to tmp of current iteration
    StructToUndirectedEdgeList* corr_undirected_edges_item = NULL;

    // Pointer to NodeIndexPair storing tmp
    NodeIndexPair* tmp = malloc(sizeof(NodeIndexPair));

    // Flag for whether or not the value was found
    bool found = false;

    // Flag used to show if we've already processed the current undirected edge
    bool processed_edge = false;

    // Pointer used to store the previously added directed edge between two nodes
    DirectedEdge* added_DE;
    DirectedEdge* this_directed_edge;

    // Add all edges to graph information
    for (int64_t i = 0; i < num_edges; i++) {
        // Haven't processed the edge yet
        processed_edge = false;
        // Create the current directed edge -------------------
        this_directed_edge = calloc(1, sizeof(DirectedEdge));
        this_directed_edge->nodes.center = center_indices[i];
        this_directed_edge->nodes.neighbor = neighbor_indices[i];
        this_directed_edge->distance = distances[i];
        this_directed_edge->index = num_directed_edges;
        this_directed_edge->image = images + (3 * i);

        // Load tmp
        memset(tmp, 0, sizeof(NodeIndexPair));
        tmp->center = center_indices[i];
        tmp->neighbor = neighbor_indices[i];

        // See if tmp is in undirected
        corr_undirected_edges_item = NULL;
        found = find_in_undirected(tmp, &undirected_edges, &corr_undirected_edges_item);

        if (!found) {
            // Never seen this edge combination before
            this_directed_edge->undirected_edge_index = num_undirected_edges;

            // Create new undirected edge
            UndirectedEdge* this_undirected_edge = malloc(sizeof(UndirectedEdge));

            directed_to_undirected(this_directed_edge, this_undirected_edge, num_undirected_edges);

            // Add this new edge information to various data structures
            create_new_undirected_edges_entry(&undirected_edges, tmp, this_undirected_edge);
            append_to_undirected_edges_list(undirected_edges_list, this_undirected_edge, &num_undirected_edges);
            add_neighbors_to_node(&nodes[center_indices[i]], neighbor_indices[i], this_directed_edge);
            append_to_directed_edges_list(directed_edges_list, this_directed_edge, &num_directed_edges);
        } else {
            // This pair of nodes has been added before. We have to check if it's the other directed edge (but pointed in
            // the different direction) OR it's another totally different undirected edge that has different image and distance

            // if found is true, then corr_undirected_edges_item points to self.undirected_edges[tmp]
            // iterate through all previously scanned undirected edges that have the same endpoints as this edge
                // if there exists an undirected edge with the same inverted image as this_undirected_edge, then add this new directed edge
                // and associate it with this undirected edge
            for (int j = 0; j < corr_undirected_edges_item->num_undirected_edges_in_group; j++) {
                // Grab the 0th directed edge associated with this undirected edge
                added_DE = directed_edges_list[((corr_undirected_edges_item->undirected_edges_list)[j]->directed_edge_indices)[0]];

                if (is_reversed_directed_edge(added_DE, this_directed_edge)) {
                    this_directed_edge->undirected_edge_index = added_DE->undirected_edge_index;
                    add_neighbors_to_node(&nodes[center_indices[i]], neighbor_indices[i], this_directed_edge);
                    append_to_directed_edges_list(directed_edges_list, this_directed_edge, &num_directed_edges);
                    append_to_directed_edge_indices((corr_undirected_edges_item->undirected_edges_list)[j], this_directed_edge->index);
                    processed_edge = true;
                    break;
                }
            }
            // There wasn't a pre-existing undirected edge that corresponds to this directed edge
            // Create a new undirected edge and process
            if (!processed_edge) {
                this_directed_edge->undirected_edge_index = num_undirected_edges;
                // Create a new undirected edge
                UndirectedEdge* this_undirected_edge = malloc(sizeof(UndirectedEdge));
                directed_to_undirected(this_directed_edge, this_undirected_edge, num_undirected_edges);
                append_to_undirected_edges_tmp(this_undirected_edge, &undirected_edges, tmp);
                append_to_undirected_edges_list(undirected_edges_list, this_undirected_edge, &num_undirected_edges);
                add_neighbors_to_node(&nodes[center_indices[i]], neighbor_indices[i], this_directed_edge);
                append_to_directed_edges_list(directed_edges_list, this_directed_edge, &num_directed_edges);
            }
        }
    }

    ReturnElems2* returned2 = malloc(sizeof(ReturnElems2));
    returned2->num_nodes = num_atoms;
    returned2->num_undirected_edges = num_undirected_edges;
    returned2->num_directed_edges = num_directed_edges;

    returned2->nodes = nodes;
    returned2->directed_edges_list = directed_edges_list;
    returned2->undirected_edges_list = undirected_edges_list;

    free(tmp);
    free_undirected_edges(&undirected_edges);

    return returned2;
}

void print_neighbors(Node* node) {
    LongToDirectedEdgeList *tmp, *neighbor;
    HASH_ITER(hh, node->neighbors, neighbor, tmp) {
        printf("C:neighboring atom: %lu\n", neighbor->key);
    }
}

void free_undirected_edges(StructToUndirectedEdgeList** undirected_edges) {
    StructToUndirectedEdgeList* current;
    StructToUndirectedEdgeList* tmp;

    HASH_ITER(hh, *undirected_edges, current, tmp) {
        HASH_DEL(*undirected_edges, current);
        free(current->undirected_edges_list);
        free(current);
    }
}

void free_LongToDirectedEdgeList_in_nodes(Node* nodes, int64_t num_nodes) {
    LongToDirectedEdgeList* current;
    LongToDirectedEdgeList* tmp;

    for (int64_t node_i = 0; node_i < num_nodes; node_i++) {
        HASH_ITER(hh, nodes[node_i].neighbors, current, tmp) {
            HASH_DEL(nodes[node_i].neighbors, current);
            free(current->directed_edges_list);
            free(current);
        }
    }
}

// Returns true if the two directed edges have images that are inverted
// NOTE: assumes that directed_edge1->center = directed_edge2->neighbor and directed_edge1->neighbor = directed_edge2->center
bool is_reversed_directed_edge(DirectedEdge* directed_edge1, DirectedEdge* directed_edge2) {
    for (int i = 0; i < 3; i++) {
        if (directed_edge1->image[i] != -1 * directed_edge2->image[i]) {
            return false;
        }
    }

    // The two directed edges should have opposing center/neighbor nodes (i.e. center-neighbor for DE1 is [0, 1] and for DE2 is [1, 0])
    // We check for that condition here
    if (directed_edge1->nodes.center != directed_edge2->nodes.neighbor) {
        return false;
    }
    if (directed_edge1->nodes.neighbor != directed_edge2->nodes.center) {
        return false;
    }
    return true;
}

// If tmp or the reverse of tmp is found in undirected_edges, True is returned and the corresponding StructToUndirectedEdgeList pointer is placed
// into found_entry. Otherwise, False is returned.
// NOTE: does not edit the *tmp
// Assumes *tmp bits have already been 0'd at padding within a struct
bool find_in_undirected(NodeIndexPair* tmp, StructToUndirectedEdgeList** undirected_edges, StructToUndirectedEdgeList** found_entry) {
    StructToUndirectedEdgeList* out_list;
    // Check tmp
    HASH_FIND(hh, *undirected_edges, tmp, sizeof(NodeIndexPair), out_list);

    if (out_list) {
        *found_entry = out_list;
        return true;
    }

    // Check tmp_rev
    NodeIndexPair tmp_rev;
    tmp_rev.center = tmp->neighbor;
    tmp_rev.neighbor = tmp->center;

    HASH_FIND(hh, *undirected_edges, &tmp_rev, sizeof(NodeIndexPair), out_list);

    if (out_list) {
        *found_entry = out_list;
        return true;
    }

    return false;
}


// Creates new entry in undirected_edges and initializes necessary arrays
void create_new_undirected_edges_entry(StructToUndirectedEdgeList** undirected_edges, NodeIndexPair* tmp, UndirectedEdge* new_undirected_edge) {
    StructToUndirectedEdgeList* new_entry = malloc(sizeof(StructToUndirectedEdgeList));
    memset(new_entry, 0, sizeof(StructToUndirectedEdgeList));

    // Set up fields within the new entry in the hashmap
    new_entry->key.center = tmp->center;
    new_entry->key.neighbor = tmp->neighbor;

    new_entry->num_undirected_edges_in_group = 1;
    new_entry->undirected_edges_list = malloc(sizeof(UndirectedEdge*));
    new_entry->undirected_edges_list[0] = new_undirected_edge;

    HASH_ADD(hh, *undirected_edges, key, sizeof(NodeIndexPair), new_entry);

}

// Appends undirected into the StructToUndirectedEdgeList entry that corresponds to tmp
// This function will first look up tmp
void append_to_undirected_edges_tmp(UndirectedEdge* undirected, StructToUndirectedEdgeList** undirected_edges, NodeIndexPair* tmp) {

    StructToUndirectedEdgeList* this_undirected_edges_item;
    find_in_undirected(tmp, undirected_edges, &this_undirected_edges_item);

    int64_t num_undirected_edges = this_undirected_edges_item->num_undirected_edges_in_group;

    // No need to worry about originally malloc'ing memory for this_undirected_edges_item->undirected_edges_list
    // this is because, we first call create_new_undirected_edges_entry for all entries. This function already mallocs for us.

    // Realloc the space to fit a new pointer to an undirected edge
    UndirectedEdge** new_list = realloc(this_undirected_edges_item->undirected_edges_list, sizeof(UndirectedEdge*) * (num_undirected_edges + 1));
    this_undirected_edges_item->undirected_edges_list = new_list;

    // Insert the undirected pointer into the newly allocated slot
    this_undirected_edges_item->undirected_edges_list[num_undirected_edges] = undirected;

    // Increase the counter for # of undirected edges
    this_undirected_edges_item->num_undirected_edges_in_group = num_undirected_edges + 1;
}


void directed_to_undirected(DirectedEdge* directed, UndirectedEdge* undirected, int64_t undirected_index) {
    // Copy over image and distance
    undirected->distance = directed->distance;
    undirected->nodes = directed->nodes;
    undirected->index = undirected_index;

    // Add a new directed_edge_index to the directed_edge_indices pointer. This should be the first
    undirected->num_directed_edges = 1;
    undirected->directed_edge_indices = malloc(sizeof(int64_t));
    undirected->directed_edge_indices[0] = directed->index;
}


void append_to_undirected_edges_list(UndirectedEdge** undirected_edges_list, UndirectedEdge* to_add, int64_t* num_undirected_edges) {
    // No need to realloc for space since our original alloc should cover everything

    // Assign value to next available position
    undirected_edges_list[*num_undirected_edges] = to_add;
    *num_undirected_edges += 1;
}

void append_to_directed_edges_list(DirectedEdge** directed_edges_list, DirectedEdge* to_add, int64_t* num_directed_edges) {
    // No need to realloc for space since our original alloc should cover everything

    // Assign value to next available position
    directed_edges_list[*num_directed_edges] = to_add;
    *num_directed_edges += 1;
}

void append_to_directed_edge_indices(UndirectedEdge* undirected_edge, int64_t directed_edge_index) {
    // TODO: don't need to realloc if we always know that there will be 2 directed edges per undirected edge. Update this later for performance boosts.
    // TODO: other random performance boost: don't pass int64_ts into function parameters, pass int64_t* instead
    undirected_edge->directed_edge_indices = realloc(undirected_edge->directed_edge_indices, sizeof(int64_t) * (undirected_edge->num_directed_edges + 1));
    undirected_edge->directed_edge_indices[undirected_edge->num_directed_edges] = directed_edge_index;
    undirected_edge->num_directed_edges += 1;
}

// If there already exists neighbor_index within the Node node, then adds directed_edge to the list of directed edges.
// If there doesn't already exist neighbor_index within the Node node, then create a new entry into the node's neighbors hashmap and add the entry
void add_neighbors_to_node(Node* node, int64_t neighbor_index, DirectedEdge* directed_edge) {
    LongToDirectedEdgeList* entry = NULL;

    // Search for the neighbor_index in our hashmap
    HASH_FIND(hh, node->neighbors, &neighbor_index, sizeof(int64_t), entry);

    if (entry) {
        // We found something, update the list within this pointer
        entry->directed_edges_list = realloc(entry->directed_edges_list, sizeof(DirectedEdge*) * (entry->num_directed_edges_in_group + 1));
        entry->directed_edges_list[entry->num_directed_edges_in_group] = directed_edge;

        entry->num_directed_edges_in_group += 1;
    } else {
        // allocate memory for entry
        entry = malloc(sizeof(LongToDirectedEdgeList));

        // The entry doesn't exist, initialize the entry and enter it into our hashmap
        entry->directed_edges_list = malloc(sizeof(DirectedEdge*));
        entry->directed_edges_list[0] = directed_edge;
        entry->key = neighbor_index;

        entry->num_directed_edges_in_group = 1;
        HASH_ADD(hh, node->neighbors, key, sizeof(int64_t), entry);

        node->num_neighbors += 1;
    }
}

// Returns a list of LongToDirectedEdgeList pointers which are entries for the neighbors of the inputted node
LongToDirectedEdgeList** get_neighbors(Node* node) {
    int64_t num_neighbors = HASH_COUNT(node->neighbors);
    LongToDirectedEdgeList** entries = malloc(sizeof(LongToDirectedEdgeList*) * num_neighbors);

    LongToDirectedEdgeList* entry;
    int64_t counter = 0;
    for (entry = node->neighbors; entry != NULL; entry = entry->hh.next) {
        entries[counter] = entry;
        counter += 1;
    }

    return entries;
}
