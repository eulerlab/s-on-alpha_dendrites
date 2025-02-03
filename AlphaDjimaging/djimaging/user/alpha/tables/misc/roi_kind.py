from abc import abstractmethod

import datajoint as dj


class RoiKindTemplate(dj.Computed):
    database = ''

    @property
    def definition(self):
        definition = """
        -> self.field_table
        ---
        roi_kind : enum('roi', 'field', 'soma', 'stack', 'other')
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.field_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    def make(self, key):
        field = (self.field_table & key).fetch1('field')

        if 'soma' in field.lower():  # Exclude old way of naming soma ROIs
            key['roi_kind'] = 'soma'
        elif 'stack' in field.lower():
            key['roi_kind'] = 'stack'
        elif 'field' in field.lower():
            key['roi_kind'] = 'field'
        elif field.lower().startswith('d'):
            key['roi_kind'] = 'roi'
        else:
            key['roi_kind'] = 'other'

        self.insert1(key, skip_duplicates=True)
