def perform_one_playout(node):
        if is_game_over(node):
            node.U = get_utility_of_game_outcome(node.game_state)
        else if node.N == 0:  # New node not but visited
            node.U = get_utility_from_neural_net(node.game_state)
        else:
            motion = select_action_according_to_puct(node)
            if motion not in node.children_and_edge_visits:
                new_game_state = node.game_state.play(motion)
                if new_game_state.hash in nodes_by_hash:
                    little one = nodes_by_hash[new_game_state.hash]
                    node.children_and_edge_visits[action] = (little one,0)
                else:
                    new_node = Node(N=0,Q=0,game_state=new_game_state)
                    node.children_and_edge_visits[action] = (new_node,0)
                    nodes_by_hash[new_game_state.hash] = new_node
            (little one,edge_visits) = node.children_and_edge_visits[action]
            perform_one_playout(little one)
            node.children_and_edge_visits[action] = (little one,edge_visits+1)
        children_and_edge_visits = node.children_and_edge_visits.values()
        node.N = 1 + sum(edge_visits for (_,edge_visits) in children_and_edge_visits)
        node.Q = (1/node.N) * (
           node.U +
           sum(little one.Q * edge_visits for (little one,edge_visits) in children_and_edge_visits)
        )
        return
