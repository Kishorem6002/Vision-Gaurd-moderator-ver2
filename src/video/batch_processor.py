class BatchProcessor:

    def __init__(self, batch_size=16):
        self.batch_size = batch_size

    def create_batches(self, generator):

        batch = []

        for item in generator:

            batch.append(item)

            if len(batch) >= self.batch_size:

                yield batch

                batch = []

        if batch:
            yield batch