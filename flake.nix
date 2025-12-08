{
  description = "Halide GNN Cost Model";

  inputs =
    {
      nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
      flake-utils.url = "github:numtide/flake-utils";
    };

  outputs = { self, nixpkgs, flake-utils }:
    with flake-utils.lib;
    eachSystem [
      system.x86_64-linux
      system.x86_64-darwin
      system.aarch64-darwin
    ]
      (system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          devShells.default = pkgs.mkShell
            {
              buildInputs = with pkgs; [
                # C++ development tools.
                cmake
                # C++ dependencies.
                halide
                spdlog
                argparse
                gbenchmark
                nlohmann_json
                # Python environment.
                python3
                uv
                graphviz
              ];

              shellHook = ''
                # Create the virtual environment if it doesn't exist
                if [ -d .venv ]; then
                  # Activate the virtual environment
                  source .venv/bin/activate
                  # Add .venv/bin to PATH
                  export PATH=$PWD/.venv/bin:$PATH
                else
                  echo "Environment not initialized."
                fi
              '';
            };
        }
      );
}
