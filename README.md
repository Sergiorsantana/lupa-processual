# Detector de comandos ocultos em processo judicial

App em Python que encontra instruções escondidas para sistemas de IA (prompt injection) dentro das peças de um processo: PDF, DOCX, TXT e imagens. Usa RAG multimodal com banco vetorial, agentes especializados e um juiz que pondera as evidências e dá o veredito de conformidade. Cada achado mostra página, área e motivo, e o resultado sai em relatório PDF para impressão.

## Como rodar

```bash
pip install -r requirements.txt
# OCR: instale o tesseract com o pacote de português (apt install tesseract-ocr tesseract-ocr-por)
# DOCX com localização visual: instale o LibreOffice (opcional)
streamlit run app.py
```

Abra o menu Indexação e clique em "Gerar e indexar documentos de exemplo" para testar sem arquivos reais. A senha do Admin é `admin` (troque com a variável `HCD_ADMIN_PASSWORD`). Os dados ficam em `data/` (mude com `HCD_DATA`).

Testes: `pytest -q`

## Menu

| Página | O que faz |
|---|---|
| Início | Números gerais e resumo das camadas |
| Indexação | Upload de documentos e imagens, processamento pelas 5 camadas |
| Dashboard | Medidor de conformidade, vereditos, severidade, categorias, mapa de calor por página, localização com caixas numeradas sobre a página, tabela filtrável e trilha dos agentes |
| Consulta RAG | Busca semântica direta no banco vetorial, com filtros por documento, modalidade e trecho oculto |
| Relatório PDF | Gera o PDF com resumo, gráficos, achados com localização, páginas destacadas, trilha e parâmetros |
| Admin | Parâmetros de todas as camadas, regras regex, sondas semânticas, coleção vetorial, reindexação, exclusão e logs |
| Arquitetura | Diagrama das camadas e regra de decisão do juiz |

## Camadas

1. **Ingestão multimodal** (`core/ingest.py`). Extrai cada trecho de texto do PDF com cor, tamanho, opacidade, modo de renderização e camada OCG. Renderiza as páginas, roda OCR, extrai imagens embutidas, metadados, XMP, anotações, campos, anexos e scripts. No DOCX lê runs com `vanish`, cor e tamanho, tabelas, caixas de texto, cabeçalhos, comentários, propriedades e texto excluído em controle de alterações. Em imagens lê metadados e bytes após o fim do arquivo.
2. **Detecção determinística** (`core/detectors.py`, `core/vision.py`). Texto branco sem fundo escuro, fonte abaixo do limite, fora da página, transparente, modo invisível (Tr 3), coberto por forma opaca, camada desligada, caracteres de largura zero, tags Unicode decodificadas, homóglifos, Base64, esteganografia LSB e texto de baixo contraste em imagem (isolado por diferença em relação ao fundo local antes do OCR).
3. **Banco vetorial e RAG** (`core/rag.py`, `core/vectorstore.py`, `core/embeddings.py`). Chunks de texto visível, texto oculto, OCR e metadados, cada um com página, bbox, modalidade e flag de oculto. ChromaDB persistente na coleção `juridico_teste` (fallback em NumPy). Embedding offline por hashing com léxico conceitual PT/EN, ou SBERT multilingue opcional.
4. **Agentes** (`core/agents.py`). O orquestrador monta o plano e executa os agentes de texto, visual, metadados e semântico. O visual confere o texto oculto contra o OCR da página renderizada. O semântico consulta o banco com as sondas e pega paráfrases. A verificação cruzada procura o mesmo comando em outras páginas.
5. **Juiz** (`core/judge.py`). Confiança do achado = peso da severidade × peso do agente × fator de visibilidade, com bônus por corroboração. Risco do documento = 1 − Π(1 − confiança). Comando crítico escondido leva direto a NÃO CONFORME. Segundo parecer por LLM é opcional e fica desligado por padrão para manter os dados locais.
6. **Apresentação** (`app.py`, `core/charts.py`, `core/viz.py`, `core/report.py`).

## Parâmetros padrão

| Parâmetro | Valor |
|---|---|
| Fonte mínima legível | 4 pt |
| Luminância tratada como branco | 0,92 |
| Chunk / sobreposição | 500 / 80 caracteres |
| Top K | 5 |
| Limiar semântico | 0,34 (embedding hashing) |
| Limite de suspeita / não conformidade | risco 0,30 / 0,60 |

Ao trocar para SBERT, recalibre o limiar semântico no Admin, porque a escala de similaridade muda.

## Limites

Linguagem comum em petições ("julgue procedente", "conceda a liminar") só vira alerta quando está escondida, para evitar falso positivo. O embedding offline reconhece paráfrases próximas do vocabulário das sondas. Texto escrito à mão ou com contraste abaixo de 3 níveis de cinza escapa do OCR. O resultado aponta indícios técnicos e pede revisão humana.
