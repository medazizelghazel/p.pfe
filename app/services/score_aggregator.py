class ScoreAggregator:
    """
    Aggregates the final course quality score.

    Formula:
        global_score =
            45% clarity
            35% engagement
            14% trainer emotion
            6% learner emotion

    Emotion labels are mapped to pedagogical scores:
        energetic_engaged -> 100
        neutral_calm      -> 80
        low_energy        -> 45
        tense_stressed    -> 35
        missing/unknown   -> 60
    """

    EMOTION_SCORE_MAP = {
        "energetic_engaged": 100.0,
        "neutral_calm": 80.0,
        "low_energy": 45.0,
        "tense_stressed": 35.0,
        "not_computed": 60.0,
        "unknown": 60.0,
        None: 60.0,
    }

    def aggregate(
        self,
        clarity_score: float,
        engagement_score: float,
        trainer_emotion: dict | None = None,
        learner_emotion: dict | None = None,
    ) -> float:
        trainer_emotion_score = self._emotion_score(trainer_emotion)

        if learner_emotion and learner_emotion.get("enabled"):
            learner_emotion_score = self._emotion_score(learner_emotion)
        else:
            learner_emotion_score = 60.0

        emotion_score = (
            0.70 * trainer_emotion_score
            + 0.30 * learner_emotion_score
        )

        global_score = (
            0.45 * self._safe_score(clarity_score)
            + 0.35 * self._safe_score(engagement_score)
            + 0.20 * emotion_score
        )

        return round(self._clamp(global_score), 2)

    def aggregate_with_details(
        self,
        clarity_score: float,
        engagement_score: float,
        trainer_emotion: dict | None = None,
        learner_emotion: dict | None = None,
    ) -> dict:
        trainer_emotion_score = self._emotion_score(trainer_emotion)

        if learner_emotion and learner_emotion.get("enabled"):
            learner_emotion_score = self._emotion_score(learner_emotion)
            learner_emotion_used = True
        else:
            learner_emotion_score = 60.0
            learner_emotion_used = False

        emotion_score = (
            0.70 * trainer_emotion_score
            + 0.30 * learner_emotion_score
        )

        global_score = self.aggregate(
            clarity_score=clarity_score,
            engagement_score=engagement_score,
            trainer_emotion=trainer_emotion,
            learner_emotion=learner_emotion,
        )

        return {
            "global_score": global_score,
            "formula": "0.45*clarity + 0.35*engagement + 0.20*emotion",
            "weights": {
                "clarity": 0.45,
                "engagement": 0.35,
                "emotion": 0.20,
                "trainer_emotion_inside_emotion": 0.70,
                "learner_emotion_inside_emotion": 0.30,
            },
            "components": {
                "clarity_score": self._safe_score(clarity_score),
                "engagement_score": self._safe_score(engagement_score),
                "trainer_emotion_score": trainer_emotion_score,
                "learner_emotion_score": learner_emotion_score,
                "emotion_score": round(emotion_score, 2),
                "learner_emotion_used": learner_emotion_used,
            },
        }

    def _emotion_score(self, emotion_result: dict | None) -> float:
        if not emotion_result:
            return 60.0

        dominant_emotion = emotion_result.get("dominant_emotion")
        score = self.EMOTION_SCORE_MAP.get(dominant_emotion, 60.0)

        return self._clamp(score)

    def _safe_score(self, value: float | int | None) -> float:
        try:
            if value is None:
                return 0.0

            return self._clamp(float(value))
        except Exception:
            return 0.0

    def _clamp(self, value: float) -> float:
        return max(0.0, min(100.0, value))