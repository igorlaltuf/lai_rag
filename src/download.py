from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import RAW_DIR, ensure_dirs, load_settings


SAMPLE_ROWS = [
    {
        "protocolo": "202600001",
        "orgao": "Ministerio da Gestao e da Inovacao",
        "data_pedido": "2026-02-10",
        "tema": "inteligencia artificial",
        "pedido": "Solicito contratos, termos de referencia e estudos tecnicos sobre uso de inteligencia artificial em atendimento ao cidadao.",
        "resposta": "Foram localizados dois contratos e um estudo tecnico preliminar. Dados pessoais e segredos comerciais foram tarjados.",
        "status": "Respondido parcialmente",
        "recurso": "Recurso pediu abertura dos anexos tarjados.",
        "decisao_recurso": "Parcialmente provido",
    },
    {
        "protocolo": "202600002",
        "orgao": "Ministerio da Justica e Seguranca Publica",
        "data_pedido": "2026-03-18",
        "tema": "reconhecimento facial",
        "pedido": "Informar quais sistemas de reconhecimento facial estao em uso, bases consultadas, fornecedores e relatorios de impacto.",
        "resposta": "O orgao informou a existencia de projeto piloto, mas negou detalhes tecnicos por seguranca publica.",
        "status": "Acesso parcialmente concedido",
        "recurso": "Recurso questionou a negativa generica de seguranca publica.",
        "decisao_recurso": "Indeferido",
    },
    {
        "protocolo": "202600003",
        "orgao": "Controladoria-Geral da Uniao",
        "data_pedido": "2026-04-04",
        "tema": "recursos LAI",
        "pedido": "Solicito estatisticas de recursos de LAI sobre contratos de tecnologia, por orgao e resultado.",
        "resposta": "Foram enviados dados agregados por orgao, mes e resultado do recurso.",
        "status": "Acesso concedido",
        "recurso": "",
        "decisao_recurso": "",
    },
    {
        "protocolo": "202600004",
        "orgao": "Ministerio da Saude",
        "data_pedido": "2026-05-20",
        "tema": "LGPD",
        "pedido": "Solicito documentos de avaliacao de impacto a protecao de dados pessoais em sistemas de saude digital.",
        "resposta": "O orgao indicou que parte dos documentos ainda esta em elaboracao e forneceu politica geral de privacidade.",
        "status": "Informacao inexistente parcial",
        "recurso": "Recurso pediu lista de sistemas ja avaliados.",
        "decisao_recurso": "Aguardando decisao",
    },
]


def discover_files(page_url: str, year: int) -> list[str]:
    response = requests.get(page_url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    urls: list[str] = []
    for link in soup.find_all("a", href=True):
        href = urljoin(page_url, link["href"])
        text = f"{link.get_text(' ')} {href}".lower()
        if str(year) not in text:
            continue
        if not re.search(r"\.(csv|zip|xml|xlsx?)(\?|$)", urlparse(href).path.lower()):
            continue
        if any(term in text for term in ["lai", "pedido", "recurso", "solicitacao", "resposta"]):
            urls.append(href)
    return sorted(set(urls))


def official_falabr_urls(year: int) -> list[str]:
    base = "https://dadosabertos-download.cgu.gov.br/FalaBR"
    return [
        f"{base}/Arquivos_FalaBR_Filtrado/Arquivos_csv_{year}.zip",
        f"{base}/Arquivos_FalaBR/Pedidos_csv_{year}.zip",
        f"{base}/Arquivos_FalaBR/Recursos_Reclamacoes_csv_{year}.zip",
    ]


def existing_urls(urls: list[str]) -> list[str]:
    available: list[str] = []
    for url in urls:
        try:
            response = requests.head(url, allow_redirects=True, timeout=30)
            if response.status_code == 200:
                available.append(url)
        except requests.RequestException:
            continue
    return available


def filename_from_url(url: str, fallback: str) -> str:
    name = Path(urlparse(url).path).name
    return name or fallback


def download_file(url: str, dest: Path) -> Path:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 512):
                if chunk:
                    fh.write(chunk)
    return dest


def write_sample(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "sample_lai_2026.csv"
    pd.DataFrame(SAMPLE_ROWS).to_csv(path, index=False)
    return path


def run(allow_sample: bool = True) -> list[Path]:
    ensure_dirs()
    settings = load_settings()
    downloaded: list[Path] = []
    try:
        urls = discover_files(settings.falabr_data_page, settings.data_year)
    except Exception as exc:
        print(f"Falha ao descobrir arquivos em {settings.falabr_data_page}: {exc}")
        urls = []
    if not urls:
        urls = existing_urls(official_falabr_urls(settings.data_year))

    for idx, url in enumerate(urls, start=1):
        name = filename_from_url(url, f"falabr_lai_{settings.data_year}_{idx}")
        dest = RAW_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            downloaded.append(dest)
            continue
        try:
            print(f"Baixando {url}")
            downloaded.append(download_file(url, dest))
        except Exception as exc:
            print(f"Falha ao baixar {url}: {exc}")

    if not downloaded and allow_sample:
        print("Nenhum arquivo publico de 2026 foi baixado; criando corpus sintetico demonstrativo.")
        downloaded.append(write_sample(RAW_DIR))
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sample", action="store_true", help="Nao cria corpus sintetico quando o download falha.")
    args = parser.parse_args()
    files = run(allow_sample=not args.no_sample)
    for path in files:
        print(path)


if __name__ == "__main__":
    main()
