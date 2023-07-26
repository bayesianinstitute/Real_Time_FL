    def setUp(self):
        # Alice
        self.alice_wallet = BtcTxStore(testnet=False, dryrun=True)
        self.alice_wif = "L18vBLrz3A5QxJ6K4bUraQQZm6BAdjuAxU83e16y3x7eiiHTApHj"
        self.alice_node_id = address_to_node_id(
            self.alice_wallet.get_address(self.alice_wif)
        )
        self.alice_dht_node = pyp2p.dht_msg.DHT(
            node_id=self.alice_node_id,
            networking=0
        )
        self.alice_storage = tempfile.mkdtemp()
        self.alice = FileTransfer(
            pyp2p.net.Net(
                net_type="direct",
                node_type="passive",
                nat_type="preserving",
                passive_port=0,
                dht_node=self.alice_dht_node,
                wan_ip="8.8.8.8",
                debug=1
            ),
            BandwidthLimit(),
            wif=self.alice_wif,
            store_config={self.alice_storage: None}
        )

        # Bob
        self.bob_wallet = BtcTxStore(testnet=False, dryrun=True)
        self.bob_wif = "L3DBWWbuL3da2x7qAmVwBpiYKjhorJuAGobecCYQMCV7tZMAnDsr"
        self.bob_node_id = address_to_node_id(
            self.bob_wallet.get_address(self.bob_wif))
        self.bob_dht_node = pyp2p.dht_msg.DHT(
            node_id=self.bob_node_id,
            networking=0
        )
        self.bob_storage = tempfile.mkdtemp()
        self.bob = FileTransfer(
            pyp2p.net.Net(
                net_type="direct",
                node_type="passive",
                nat_type="preserving",
                passive_port=0,
                dht_node=self.bob_dht_node,
                wan_ip="8.8.8.8",
                debug=1
            ),
            BandwidthLimit(),
            wif=self.bob_wif,
            store_config={self.bob_storage: None}
        )

        # Accept all transfers.
        def accept_handler(contract_id, src_unl, data_id, file_size):
            return 1

        # Add accept handler.
        self.alice.handlers["accept"].add(accept_handler)
        self.bob.handlers["accept"].add(accept_handler)

        # Link DHT nodes.
        self.alice_dht_node.add_relay_link(self.bob_dht_node)
        self.bob_dht_node.add_relay_link(self.alice_dht_node)

        # Bypass sending messages for client.
        def send_msg(dict_obj, unl):
            print("Skipped sending message in test")
            print(dict_obj)
            print(unl)

        # Install send msg hooks.
        self.alice.send_msg = send_msg
        self.bob.send_msg = send_msg

        # Bypass sending relay messages for clients.
        def relay_msg(node_id, msg):
            print("Skipping relay message in test")
            print(node_id)
            print(msg)

        # Install relay msg hooks.
        if self.alice.net.dht_node is not None:
            self.alice.net.dht_node.relay_message = relay_msg

        if self.bob.net.dht_node is not None:
            self.bob.net.dht_node.relay_message = relay_msg

        # Bypass UNL.connect for clients.
        def unl_connect(their_unl, events, force_master=1, hairpin=1,
                        nonce="0" * 64):
            print("Skipping UNL.connect!")
            print("Their unl = ")
            print(their_unl)
            print("Events = ")
            print(events)
            print("Force master = ")
            print(force_master)
            print("Hairpin = ")
            print(hairpin)
            print("Nonce = ")
            print(nonce)

        # Install UNL connect hooks.
        self.alice.net.unl.connect = unl_connect
        self.bob.net.unl.connect = unl_connect

        # Record syn.
        data_id = u"5feceb66ffc86f38d952786c6d696c79"
        data_id += u"c2dbc239dd4e91b46729d73a27fb57e9"
        self.syn = OrderedDict([
            (u"status", u"SYN"),
            (u"data_id", data_id),
            (u"file_size", 100),
            (u"host_unl", self.alice.net.unl.value),
            (u"dest_unl", self.bob.net.unl.value),
            (u"src_unl", self.alice.net.unl.value)
        ])
