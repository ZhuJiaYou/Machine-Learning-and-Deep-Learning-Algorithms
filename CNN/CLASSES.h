/*****************************************************************************/
/*                                                                           */
/*                   Definitions of classes or structs                       */
/*                                                                           */
/*****************************************************************************/

/* Fundamental structure of a model */
typedef struct {
	int n_layers;  // Number of layers, not include final
	int layer_type[MAX_LAYERS];  // Each entry (input to final) is TYPE_? in CONST.h
	int depth[MAX_LAYERS];  // Number of hidden neurons if fully connected, else number of slices
	int HalfWidH[MAX_LAYERS];  // Horizontal half width looking back to prior layer
	int HalfWidV[MAX_LAYERS];  // And vertical
	int padH[MAX_LAYERS];  // Horizontal padding, should not exceed half width
	int padV[MAX_LAYERS];  // And vertical
	int stridH[MAX_LAYERS];  // Horizontal stride
	int stridV[MAX_LAYERS];  // And vertical
	int PoolWidH[MAX_LAYERS];  // Horizontal pooling width looking back to prior layer
	int PoolWidV[MAX_LAYERS];  // And vertical
} ARCHITECTURE;


/* Training parameters */
typedef struct {
	int max_batch;  // Divide the training set into subsets for CUDA timeouts; this is max size of a subset
	int max_hid_grad;  // Maximum number of hidden neurons per CUDA launch; pervents timeout error and lowers memory use
	int max_mem_grad;  // Maximum CONV working memory(MB) per CUDA launch; pervents timeout error and lowers memory use
	int anneal_iters;  // Supervised anneal iters
	double anneal_rng;  // Starting range for annealing
	int maxits;  // Max iterations for traning supervised section
	double tol;  // Convergence tolerance for training supervised section
	double wpen;  // Weight penalty (should be very small)
	// These are set in READ_SERIES.cpp and copied to model during training
	int class_type;  // 1=split at zero; 2=split at median; 3=split at .33 and .67 quantiles; READ_SERIES.cpp sets, MODEL.cpp uses
	double median;
	double quantile_33;
	double quantile_67;
} TRAIN_PARAMS;


/* CUDA timers */
typedef struct {
	int ncall_weights;
	int weights;
	int ncalls_act[MAX_LAYERS + 1];
	int act[MAX_LAYERS + 1];
	int ncalls_softmax;
	int softmax;
	int ncalls_ll;
	int ll;
	int ncalls_delta[MAX_LAYERS + 1];
	int grad[MAX_LAYERS + 1];
	int ncalls_movedelta;
	int movedelta;
	int ncalls_fetchgrad;
	int fetchgrad;
} CUDA_TIMERS;



