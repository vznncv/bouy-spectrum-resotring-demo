def pytest_configure():
    import logging
    logging.basicConfig(level=logging.INFO)

    # add src folder
    try:
        import spectrum_processing_1d
    except ImportError:
        import sys
        from os.path import abspath, dirname, join
        sys.path.append(abspath(join(dirname(__file__), '..', 'src')))
