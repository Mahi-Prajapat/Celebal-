class Node:
    """Class to represent a node in a singly linked list."""
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    """Class to manage the singly linked list."""
    def __init__(self):
        self.head = None

    def add_node(self, data):
        """Add a node to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node

    def print_list(self):
        """Print the entire list."""
        current_node = self.head
        if not current_node:
            print("The list is empty.")
            return
        while current_node:
            print(current_node.data, end=" -> ")
            current_node = current_node.next
        print("None")

    def delete_nth_node(self, n):
        """Delete the nth node (1-based index) from the list."""
        if not self.head:
            raise Exception("Cannot delete from an empty list.")
        
        if n < 1:
            raise Exception("Index must be a positive integer.")
        
        current_node = self.head
        
        # If the head needs to be removed
        if n == 1:
            self.head = current_node.next
            return
        
        # Find the previous node of the node to be deleted
        for _ in range(n - 2):
            current_node = current_node.next
            if current_node is None:
                raise Exception("Index out of range.")
        
        # If the current node is None, the index is out of range
        if current_node.next is None:
            raise Exception("Index out of range.")
        
        # Node to be deleted
        current_node.next = current_node.next.next


# Testing the LinkedList implementation
if __name__ == "__main__":
    linked_list = LinkedList()
    
    # Adding nodes to the list
    linked_list.add_node(1)
    linked_list.add_node(2)
    linked_list.add_node(3)
    linked_list.add_node(4)
    
    print("Initial list:")
    linked_list.print_list()
    
    # Deleting the 2nd node
    print("Deleting the 2nd node:")
    linked_list.delete_nth_node(2)
    linked_list.print_list()
    
    # Deleting the 1st node
    print("Deleting the 1st node:")
    linked_list.delete_nth_node(1)
    linked_list.print_list()
    
    # Attempting to delete a node from an empty list
    print("Attempting to delete from an empty list:")
    linked_list.delete_nth_node(1)  # This will raise an exception
