def test_public_imports():
    import compresstab
    from compresstab.pipeline import CompressTabPipeline

    assert compresstab.CompressTabPipeline is CompressTabPipeline
    assert set(compresstab.__all__) == {
        "CompressedTable", "CompressTab", "CompressTabPipeline",
        "ExactTokenCounter", "InferenceResult", "JsonLLMClient", "PipelineConfig",
    }
    for name in compresstab.__all__:
        assert getattr(compresstab, name) is not None
