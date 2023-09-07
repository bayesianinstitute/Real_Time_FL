from ipfshttpclient import connect

class IPFSInfuraConnection:
    def __init__(self, project_id, project_secret):
        self.project_id = project_id
        self.project_secret = project_secret
        self.ipfs = connect(
            '/ip4/ipfs.infura.io/tcp/5001/https',
            headers={"Authorization": f"Basic {project_id}:{project_secret}"}
        )

    def add_to_ipfs(self, data):
        res = self.ipfs.add_bytes(data)
        print(f"File added to IPFS with CID: {res['Hash']}")

    def get_from_ipfs(self, cid):
        data = self.ipfs.cat(cid)
        # Handle the data (e.g., save it to a file)
        return data

if __name__ == '__main__':
    # Replace with your Infura project ID and secret
    project_id = 'YOUR_PROJECT_ID'
    project_secret = 'YOUR_PROJECT_SECRET'

    # Create an instance of the IPFSInfuraConnection class
    ipfs_connection = IPFSInfuraConnection(project_id, project_secret)

    # Example usage
    ipfs_connection.add_to_ipfs(b'Hello, IPFS!')

