from abc import abstractmethod

import datajoint as dj


class CellTagsTemplate(dj.Manual):
    database = ''

    @property
    def definition(self):
        definition = """
        # Cell tags
        -> self.experiment_table
        ---
        cell_tag: varchar(191)
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.experiment_table().proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def experiment_table(self):
        pass
