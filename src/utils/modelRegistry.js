const I18N_BUILD_DIR = "../../prep/onnx/build/i18n";

function assetUrl(filename) {
  return new URL(`${I18N_BUILD_DIR}/${filename}`, import.meta.url).href;
}

export const LANGUAGE_MODEL_CONFIG = {
  en: {
    attention: {
      localModelId: "minilm_en",
    },
    word2vec: {
      embedUrl: assetUrl("word2vec_en_embed.onnx"),
      vectorKnnUrl: assetUrl("word2vec_en_vector_knn.onnx"),
      vocabUrl: assetUrl("word2vec_en_vocab.json"),
      metadataUrl: assetUrl("word2vec_en_metadata.json"),
    },
  },
  fr: {
    attention: {
      localModelId: "camembert_fr",
    },
    word2vec: {
      embedUrl: assetUrl("word2vec_fr_embed.onnx"),
      vectorKnnUrl: assetUrl("word2vec_fr_vector_knn.onnx"),
      vocabUrl: assetUrl("word2vec_fr_vocab.json"),
      metadataUrl: assetUrl("word2vec_fr_metadata.json"),
    },
  },
  de: {
    attention: {
      localModelId: "bert_de",
    },
    word2vec: {
      embedUrl: assetUrl("word2vec_de_embed.onnx"),
      vectorKnnUrl: assetUrl("word2vec_de_vector_knn.onnx"),
      vocabUrl: assetUrl("word2vec_de_vocab.json"),
      metadataUrl: assetUrl("word2vec_de_metadata.json"),
    },
  },
  it: {
    attention: {
      localModelId: "bert_it",
    },
    word2vec: {
      embedUrl: assetUrl("word2vec_it_embed.onnx"),
      vectorKnnUrl: assetUrl("word2vec_it_vector_knn.onnx"),
      vocabUrl: assetUrl("word2vec_it_vocab.json"),
      metadataUrl: assetUrl("word2vec_it_metadata.json"),
    },
  },
};

export function resolveLanguage(language) {
  return LANGUAGE_MODEL_CONFIG[language] ? language : "en";
}

export function getWord2VecModelConfig(language) {
  return LANGUAGE_MODEL_CONFIG[resolveLanguage(language)].word2vec;
}

export function getAttentionModelConfig(language) {
  return LANGUAGE_MODEL_CONFIG[resolveLanguage(language)].attention;
}
