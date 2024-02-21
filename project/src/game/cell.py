from enum import Enum


class CellState(Enum):
    HIDDEN = 0
    REVEALED = 1
    FLAGGED = 2


class Cell:

    def __init__(self, state: CellState, value: int, is_mine: bool) -> None:
        self.state = state
        self.value = value
        self.is_mine = is_mine

    # Requests
    def get_state(self) -> CellState:
        """Get the state of the cell"""
        return self.state

    def get_value(self) -> int:
        """Get the value of the cell"""
        return self.value

    def get_is_mine(self) -> bool:
        """Get whether the cell is a mine or not"""
        return self.is_mine

    # Commands
    def set_state(self, state: CellState) -> None:
        """Set the state of the cell"""
        self.state = state

    def set_value(self, value: int) -> None:
        """Set the value of the cell"""
        self.value = value

    def set_is_mine(self, is_mine: bool) -> None:
        """Set whether the cell is a mine or not"""
        self.is_mine = is_mine

    def reveal(self) -> None:
        """Reveal the cell if it is hidden"""
        self.state = CellState.REVEALED

    def flag(self) -> None:
        """Flag the cell if it is hidden"""
        self.state = CellState.FLAGGED

    def __repr__(self) -> str:
        return f"Cell({self.state}, {self.value}, {self.is_mine})"
