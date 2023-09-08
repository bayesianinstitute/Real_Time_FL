/**
 * Use this file to configure your truffle project. It's seeded with some
 * common settings for different networks and features like migrations,
 * compilation and testing. Uncomment the ones you need or modify
 * them to suit your project as necessary.
 *
 * More information about configuration can be found at:
 * 
 * https://trufflesuite.com/docs/truffle/reference/configuration
 *
 * To deploy via Infura you'll need a wallet provider (like @truffle/hdwallet-provider)
 * to sign your transactions before they're sent to a remote public node. Infura accounts
 * are available for free at: infura.io/register.
 *
 * You'll also need a mnemonic - the twelve word phrase the wallet uses to generate
 * public/private key pairs. If you're publishing your code to GitHub make sure you load this
 * phrase from a file you've .gitignored so it doesn't accidentally become public.
 *
 */

const HDWalletProvider = require('@truffle/hdwallet-provider');
const fs = require('fs');
const dotenv = require('dotenv');

// Load environment variables from .env file
dotenv.config();

// Access the mnemonic from the environment variables
const mnemonic = process.env.MNEMONIC;

// const projectApi=process.env.PROJECTAPI;

// Check if the mnemonic is defined
if (!mnemonic ) {
  console.error("Mnemonic or projectApi not found in .env file.");
  process.exit(1);
}


  module.exports = {
    networks: {
      development: {
        host: "127.0.0.1",     // Localhost (default: none)
        port: 7545,            // Standard G port (default: none)
        network_id: "*",       // Any network (default: none)
      },
  
  
      // sepolia: {
      //   provider: () => new HDWalletProvider(mnemonic , projectApi),
      //   network_id: 11155111,       // Goerli's id
      //   confirmations: 2,    // # of confirmations to wait between deployments. (default: 0)
      //   timeoutBlocks: 200,  // # of blocks before a deployment times out  (minimum/default: 50)
      //   skipDryRun: true     // Skip dry run before migrations? (default: false for public nets )
      // },
  
      // godetest: {
      //   provider: () => new HDWalletProvider(mnemonic, `https://rpctest.godechain.com`),
      //   network_id: 5566,
      //   // confirmations: 10,
      //   // timeoutBlocks: 200,
      //   skipDryRun: true
      // },
      // gode: {
      //   provider: () => new HDWalletProvider(mnemonic, `https://rpc.godechain.com`),
      //   network_id: 5500,
      //   confirmations: 10,
      //   timeoutBlocks: 200,
      //   skipDryRun: true
      // },
    },
  

  // Set default mocha options here, use special reporters etc.
  mocha: {
    // timeout: 100000
  },

  // Configure your compilers
  compilers: {
    solc: {
      version: "0.8.13",      // Fetch exact version from solc-bin (default: truffle's version)
      // docker: true,        // Use "0.5.1" you've installed locally with docker (default: false)
      // settings: {          // See the solidity docs for advice about optimization and evmVersion
      //  optimizer: {
      //    enabled: false,
      //    runs: 200
      //  },
      //  evmVersion: "byzantium"
      // }
    }
  },

  // Truffle DB is currently disabled by default; to enable it, change enabled:
  // false to enabled: true. The default storage location can also be
  // overridden by specifying the adapter settings, as shown in the commented code below.
  //
  // NOTE: It is not possible to migrate your contracts to truffle DB and you should
  // make a backup of your artifacts to a safe location before enabling this feature.
  //
  // After you backed up your artifacts you can utilize db by running migrate as follows:
  // $ truffle migrate --reset --compile-all
  //
  // db: {
    // enabled: false,
    // host: "127.0.0.1",
    // adapter: {
    //   name: "sqlite",
    //   settings: {
    //     directory: ".db"
    //   }
    // }
  // }
};
