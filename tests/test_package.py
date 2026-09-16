def test_backend_package_imports():
    import backend.app
    import backend.pipeline
    import backend.providers

    assert backend.pipeline.__name__ == "backend.pipeline"
