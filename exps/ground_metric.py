import torch


def isnan(x):
    return x != x
    
class GroundMetric:
    """
        Ground Metric object for Wasserstein computations:

    """
    groud_metrics_obj = None
    def __init__(self, args):
        self.args = args
        # self.ground_metric_type = params.ground_metric

    def _clip(self, ground_metric_matrix):
        if self.args.debug:
            print("before clipping", ground_metric_matrix.data)

        percent_clipped = (float((ground_metric_matrix >= self.args.reg * self.args.clip_max).long().sum().data) \
                           / ground_metric_matrix.numel()) * 100
        print("percent_clipped is (assumes clip_min = 0) ", percent_clipped)
        setattr(self.args, 'percent_clipped', percent_clipped)
        # will keep the M' = M/reg in range clip_min and clip_max
        ground_metric_matrix.clamp_(min=self.args.reg * self.args.clip_min,
                                             max=self.args.reg * self.args.clip_max)
        if self.args.debug:
            print("after clipping", ground_metric_matrix.data)
        return ground_metric_matrix

    def _normalize(self, ground_metric_matrix):

        if self.args.ground_metric_normalize == "log":
            print("Normalizing the ground metric by log1p ")
            ground_metric_matrix = torch.log1p(ground_metric_matrix)
        elif self.args.ground_metric_normalize == "max":
            print("Normalizing by max of ground metric and which is ", ground_metric_matrix.max())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.max()
        elif self.args.ground_metric_normalize == "median":
            print("Normalizing by median of ground metric and which is ", ground_metric_matrix.median())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.median()
        elif self.args.ground_metric_normalize == "mean":
            print("Normalizing by mean of ground metric and which is ", ground_metric_matrix.mean())
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.mean()
        elif self.args.ground_metric_normalize == "none":
            return ground_metric_matrix
        else:
            raise NotImplementedError

        return ground_metric_matrix

    def _sanity_check(self, ground_metric_matrix):
        assert not (ground_metric_matrix < 0).any()
        assert not (isnan(ground_metric_matrix).any())

    def _cost_matrix_xy(self, x, y, p=2):
        # TODO: Use this to guarantee reproducibility of previous results and then move onto better way
        "Returns the matrix of $|x_i-y_j|^p$."
        # print("GroundMetric -> _cost_matrix_xy")
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
        c = c ** (1/2)
        return c


    def _pairwise_distances(self, x, y):
        '''
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)


        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        dist = torch.clamp(dist, min=0.0)

        dist = dist ** (1/2)

        return dist

    def _get_euclidean(self, coordinates, other_coordinates):
        # TODO: Replace by torch.pdist (which is said to be much more memory efficient)
        if hasattr(self.args, 'mem_efficient') and self.args.mem_efficient:
            print("use mem efficient way to calculate ground metric")
            matrix = self._pairwise_distances(coordinates, other_coordinates)
        else:
            matrix = self._cost_matrix_xy(coordinates, other_coordinates)

        return matrix

    def _normed_vecs(self, vecs, eps=1e-9):
        norms = torch.norm(vecs, dim=-1, keepdim=True)
        print("stats of vecs are: mean {}, min {}, max {}, std {}".format(
            norms.mean(), norms.min(), norms.max(), norms.std()
        ))
        return vecs / (norms + eps)

    def _get_cosine(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = coordinates / torch.norm(coordinates, dim=1, keepdim=True)
            matrix = 1 - matrix @ matrix.t()
        else:
            matrix = 1 - torch.div(
                coordinates @ other_coordinates.t(),
                torch.norm(coordinates, dim=1).view(-1, 1) @ torch.norm(other_coordinates, dim=1).view(1, -1)
            )
        return matrix.clamp_(min=0)

    def _get_angular(self, coordinates, other_coordinates=None):
        pass

    def get_metric(self, coordinates, other_coordinates=None):
        get_metric_map = {
            'euclidean': self._get_euclidean,
            'cosine': self._get_cosine,
            'angular': self._get_angular,
        }
        # print("ground_metric:", self.args.ground_metric)
        return get_metric_map[self.args.ground_metric](coordinates, other_coordinates)
    
    def process(self, coordinates, other_coordinates=None):
        return GroundMetric.PROCESS(self.args, coordinates, other_coordinates, self)

    @staticmethod
    def PROCESS(args, coordinates, other_coordinates, groud_metrics_obj=None):
        if groud_metrics_obj == None:
            if GroundMetric.groud_metrics_obj == None:
                groud_metrics_obj = GroundMetric(args)
                GroundMetric.groud_metrics_obj = groud_metrics_obj
            else:
                groud_metrics_obj = GroundMetric.groud_metrics_obj

        # print('Processing the coordinates to form ground_metric')
        if hasattr(args, 'normalize_coords') and args.normalize_coords:
            print("metrics PROCESS: normalizing coords to unit norm")
            coordinates = groud_metrics_obj._normed_vecs(coordinates)
            other_coordinates = groud_metrics_obj._normed_vecs(other_coordinates)

        ground_metric_matrix = groud_metrics_obj.get_metric(coordinates, other_coordinates)

        if args.debug:
            print("coordinates is ", coordinates)
            if other_coordinates is not None:
                print("other_coordinates is ", other_coordinates)
            print("ground_metric_matrix is ", ground_metric_matrix)

        groud_metrics_obj._sanity_check(ground_metric_matrix)

        ground_metric_matrix = groud_metrics_obj._normalize(ground_metric_matrix)

        groud_metrics_obj._sanity_check(ground_metric_matrix)

        if args.clip_gm:
            ground_metric_matrix = groud_metrics_obj._clip(ground_metric_matrix)

        groud_metrics_obj._sanity_check(ground_metric_matrix)

        if args.debug:
            print("ground_metric_matrix at the end is ", ground_metric_matrix)

        return ground_metric_matrix

