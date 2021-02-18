import numpy as np


class Evaluator:
    ''' Class for the evaluation of information retrieval systems.

    The class allows for the computation of:
    * the (recall, precision) points for a single query, after simple
    interpolation or 11-pt interpolation,
    * the average precision (AP) for a single query,
    * the averaged 11-pt interpolated (recall, precision) points for all
    queries,
    * the mean average precision (mAP) computed over all queries.

    Each Evaluator object should be build for the evaluation of a given run
    over a given dataset. Upon construction, one should provide:
    * the search results of the run over the dataset,
    * the groundtruth of the dataset.
    Search results and groundtruth should be provided as dictionaries with
    the following structure:
    { 'groundtruth':
        [{'id':id_1, 'relevant':[rel_11, rel12, rel13...]},
         {'id':id_2, 'relevant':[rel_21, rel22, rel23...]},
         ...
        ]
    }
    with id_i the id of a query and [rel_i1, rel_i2, rel_i3...] the list of
    relevant / retrieved documents for this query. This list must be sorted
    by estimated relevance for retrieved documents. The root element
    ('groundtruth') may be different (e.g. 'run', 'retrieved'...) for the
    dictionary of search results.

    The evaluation measures and limit cases (absence of relevant documents
    or retrieved documents) are computed in the same way as in trec_eval.
    '''

    def __init__(self, retrieved, relevant):
        ''' Constructor for an Evaluator object.

        Builds an Evaluator object for a run given the lists of
        relevant documents and retrieved documents for each query.
        These lists should follow the dictionary format described in
        the documentation of the class.

        :param retrieved: List of retrieved documents for each query,
        sorted by estimated relevance.
        :param relevant: List of relevant documents (groundtruth)
        for each query.
        :type retrieved: List of Dict
        :type relevant: List of Dict
        '''
        self._retrieved = self._flatten_json_qrel(retrieved, root=list(retrieved.keys())[0])
        self._relevant = self._flatten_json_qrel(relevant)

    def _flatten_json_qrel(self, json_qrel, root='groundtruth'):
        return {item['id']: item['relevant'] for item in json_qrel[root]}

    def _interpolate_11pts(self, rp_points):
        rp_11pts = []
        recalls = np.array([rp[0] for rp in rp_points])
        precisions = np.array([rp[1] for rp in rp_points])

        for recall_cutoff in np.arange(0., 1.01, .1):
            if np.count_nonzero(recalls >= recall_cutoff) > 0:
                rp_11pts.append((recall_cutoff, np.max(precisions[recalls >= recall_cutoff])))
            else:
                rp_11pts.append((recall_cutoff, 0.))
        return rp_11pts

    def _evaluate_query_pr(self, retrieved, relevant, interpolation_11pts=True):
        # if no grountruth is available
        if relevant is None or len(relevant) == 0:
            return None

        # if nothing was retrieved
        if retrieved is None or len(retrieved) == 0:
            if interpolation_11pts:
                return [(r, 0.) for r in np.arange(0., 1.01, .1)]
            else:
                return [(0., 0.)]

        # now we can work
        rp_points = {0.: (0., 0.)}
        tps = 0
        for i, retrieved_doc_id in enumerate(retrieved):
            if retrieved_doc_id in relevant:
                tps += 1
            recall = float(tps) / float(len(relevant))
            precision = float(tps) / float(i + 1)
            if recall in rp_points:
                # keep best precision for given recall
                if precision > rp_points[recall][1]:
                    rp_points[recall] = (recall, precision)
            else:
                rp_points[recall] = (recall, precision)

        rp_points = [rp_points[r] for r in sorted(rp_points.keys())]

        # fix P@0
        if len(rp_points) > 1:
            rp_points[0] = (0., rp_points[1][1])

        if interpolation_11pts:
            rp_points = self._interpolate_11pts(rp_points)

        return rp_points

    def _evaluate_query_ap(self, retrieved, relevant):
        # if no grountruth is available
        if relevant is None or len(relevant) == 0:
            return np.nan

        # if nothing was retrieved
        if retrieved is None or len(retrieved) == 0:
            return 0.

        # now we can work
        ap = 0.
        tps = 0
        for i, retrieved_doc_id in enumerate(retrieved):
            if retrieved_doc_id in relevant:
                tps += 1
                ap += float(tps) / float(i + 1)

        return ap / len(relevant)

    ''' Compute the interpolated (recall, precision) points for a given
    query.

    :param query_id: ID of the query to be evaluated.
    :param interpolation_11pts: if True, 11-pt interpolation is used.
    Otherwise, regular interpolation is used (Default: True).
    :type query_id: integer or string (depending on the data initially
    provided)
    :type interpolation_11pts: Bool

    :return: (recall, precision) points
    :rtype: list of (float, float) tuples
    '''

    def evaluate_query_pr_points(self, query_id, interpolation_11pts=True):
        return self._evaluate_query_pr(self._retrieved.get(query_id), self._relevant.get(query_id), interpolation_11pts)

    ''' Compute the average precision (AP) for a given query.

    :param query_id: ID of the query to be evaluated.
    :type query_id: integer or string (depending on the data intially
    provided).

    :return: The AP for the query.
    :rtype: float
    '''

    def evaluate_query_ap(self, query_id, interpolation_11pts=True):
        return self._evaluate_query_ap(self._retrieved.get(query_id), self._relevant.get(query_id))

    ''' Compute the 11-pt interpolated (recall, precision) points averaged
    over the queries of the run.

    :return: averaged (recall, precision) points
    :rtype: list of (float, float) tuples
    '''

    def evaluate_pr_points(self):
        precisions = []
        for i, qid in enumerate(self._relevant.keys()):
            q_pr = self._evaluate_query_pr(self._retrieved.get(qid), self._relevant[qid], interpolation_11pts=True)
            if q_pr is not None:
                precisions.append([pr[1] for pr in q_pr])
        return list(zip(np.arange(0., 1.01, .1), np.mean(np.array(precisions), axis=0)))

    ''' Compute the mean average precision (mAP) over the set of queries of
    the run.

    :return: The mAP of the run.
    :rtype: float
    '''

    def evaluate_map(self):
        aps = np.array(
            [self._evaluate_query_ap(self._retrieved.get(qid), self._relevant[qid]) for qid in self._relevant])
        return np.mean(aps[~np.isnan(aps)])
