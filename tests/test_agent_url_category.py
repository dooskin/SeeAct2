from seeact.agent import SeeActAgent
from seeact.runner import _domain_from_url


def test_categorize_url_flags_collection_and_pdp():
    coll, pdp = SeeActAgent._categorize_url("https://shop.example.com/collections/hijabs?view=grid")
    assert coll is True
    assert pdp is False

    coll, pdp = SeeActAgent._categorize_url("https://shop.example.com/products/silk-hijab")
    assert coll is False
    assert pdp is True

    coll, pdp = SeeActAgent._categorize_url("https://shop.example.com/pages/about")
    assert coll is False
    assert pdp is False


def test_domain_from_url_helper():
    assert _domain_from_url("https://www.example.com/shop") == "example.com"
    assert _domain_from_url("example.com/products") == "example.com"
    assert _domain_from_url("invalid") == "invalid"
