class AwsS3CosHelper:
    """Compatibility stub for upstream datasets that reference cloud geometry."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Remote S3/COS geometry is unavailable in the open-source local-data pipeline."
        )
