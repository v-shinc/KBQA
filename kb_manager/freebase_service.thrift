
service FreebaseService{
    list<list<list<string>>> get_subgraph(1:string subject)
    list<list<string>> get_relations(1:string subject)
    list<list<string>> get_one_hop_paths(1:string subject)
}