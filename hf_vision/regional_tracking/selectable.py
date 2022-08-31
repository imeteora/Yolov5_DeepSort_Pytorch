from hf_vision.regional_tracking.line_interset_util import Point


class Selectable:
    _selected = False

    @property
    def is_selected(self) -> bool:
        return self._selected

    @is_selected.setter
    def is_selected(self, new_value):
        self._selected = new_value

    def hit_test(self, pt: Point) -> bool:
        assert (False, 'must be implemented in sub-classes.')
        pass
