let
    pkgs = import <unstable> { };
    lib = pkgs.lib;
    stdenv = pkgs.stdenv;
    poetryPatched = pkgs.python39Packages.poetry.overridePythonAttrs {
        preCheck = lib.optionalString (stdenv.isDarwin && stdenv.isAarch64) ''
            # https://github.com/python/cpython/issues/74570#issuecomment-1093748531
            export no_proxy='*';
        '';
    };
in pkgs.mkShell {
    buildInputs = with pkgs; [
        python39
        # TODO this should be python39Packages.poetry but it is broken on M1 MacOS due to 1.2 release
        poetryPatched
    ];
    shellHook =
    ''
        export PYTHONPATH=$PYTHONPATH:$(pwd)
        alias poe="poetry run poe";
        alias rge="rg -g '!{**/migrations/*.py,**/node_modules/**,**/*.json,**/*.csv}'";
        alias rger="rg -g '!{**/migrations/*.py,**/node_modules/**,**/*.json,**/*.csv,**/*.R}'";
    '';

}