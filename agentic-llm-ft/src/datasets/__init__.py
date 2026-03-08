from __future__ import annotations


class Dataset:
    def __init__(self, rows: list[dict]):
        self._rows = rows
        keys = set()
        for row in rows:
            keys.update(row.keys())
        self.features = {key: object for key in keys}

    @classmethod
    def from_list(cls, rows: list[dict]):
        return cls(rows)

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)
