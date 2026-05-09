class ScoreAggregator:
    def aggregate(self, clarity_score: float, engagement_score: float) -> float:
        global_score = (0.6 * clarity_score) + (0.4 * engagement_score)
        return round(global_score, 2)