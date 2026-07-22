from layercake.moonshot_final import main


def test_final_cli_requires_a_subcommand():
    try:
        main([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("CLI unexpectedly accepted no subcommand")
