from pydantic import BaseModel
from typing import List


class Kural(BaseModel):
    number: str
    tamil: str


class KuralEnglish(BaseModel):
    number: str
    english: str


class Explanation(BaseModel):
    English: str
    Tamil: str


class Paal(BaseModel):
    English: str
    Tamil: str


class Adhigaram(BaseModel):
    Chapter_No: str
    English: str
    Tamil: str


class Story(BaseModel):
    English: str
    Tamil: str


class MatchingKural(BaseModel):
    number: str
    English: str
    Tamil: str


class ThirukuralResponse(BaseModel):
    Kural: Kural
    Kural_English: KuralEnglish
    Explanation: Explanation
    Paal: Paal
    Adhigaram: Adhigaram
    Story: Story
    Other_Matching_Kurals: List[MatchingKural]
