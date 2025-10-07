def parse_optimizer(optimizer_str: str, **kwargs):
    if "Adam" in optimizer_str:
        return {
            "name": "adam",
            "params": kwargs,
        }
    elif "SGD" in optimizer_str:
        return {
            "name": "sgd",
            "params": kwargs,
        }
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_str}")
