"""Native OME-NGFF (OME-Zarr) serving for Ultra.

A standalone, stateless reader+render layer for OME-Zarr multiscale stores, served by
the `ngff-service` microservice (see ``ngff.service`` / ``ultra_deepagents.ngff_service``).
It reads chunks lazily via zarr-python — never the whole array — so a 1 TB store serves
only the chunks a viewport needs at the chosen multiscale level. No libbioimage, no Java.
"""

from ultra_deepagents.ngff.reader import NgffError, NgffImage, open_ngff

__all__ = ["NgffImage", "NgffError", "open_ngff"]
