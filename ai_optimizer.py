import torch
import redis
import json
from torch_geometric.nn import MetaPath2Vec

r = redis.Redis(host='192.168.1.102', port=6379)

class CyberDefenseAI(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.metapath = MetaPath2Vec(
            edge_index_dict={('node', 'to', 'node'): torch.tensor([[0,1],[1,2]])},
            embedding_dim=128,
            metapath=[('node', 'to', 'node')],
            walk_length=5,
            context_size=3
        )
        self.classifier = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=128, nhead=8),
            num_layers=3
        )
    
    def forward(self, x):
        embeddings = self.metapath()
        return self.classifier(embeddings)

def adaptive_mitigation():
    model = CyberDefenseAI().load_state_dict(torch.load('cyber_defense.pt'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    pubsub = r.pubsub()
    pubsub.subscribe('alerts')
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            alert = json.loads(message['data'])
            mitigation = model(torch.tensor(alert['features']))
            
            if mitigation.argmax() == 0:
                r.publish('mitigation', json.dumps({
                    'action': 'block',
                    'target': alert['source'],
                    'duration': '300s'
                }))
            elif mitigation.argmax() == 1:
                r.publish('mitigation', json.dumps({
                    'action': 'throttle',
                    'limit': '1mbps',
                    'target': alert['source']
                }))
