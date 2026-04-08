from bq.data_service.controllers import resource_query


class _DummyField:
    def __eq__(self, other):
        return ("eq", other)

    def like(self, other):
        return ("like", other)


class _DummyType:
    resource_name = _DummyField()


class _DummyQuery:
    def __init__(self):
        self.filters = []

    def filter(self, *exprs):
        self.filters.extend(exprs)
        return self


def test_prepare_attributes_treats_string_as_single_value():
    query = _DummyQuery()
    attribs = {"name": "qa_0218014930_a"}

    resource_query.prepare_attributes(query, _DummyType, attribs)

    assert query.filters == [("eq", "qa_0218014930_a")]
