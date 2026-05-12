"""Semantic categories used by the invoice normalization scripts.

This lightweight module avoids importing the full GNN/visualization training
script when only the category vocabulary is needed.
"""

INVOICE_CATEGORIES = {
    "cod_cliente": ["codigo do cliente", "cod cliente", "customer code", "numero cliente"],
    "conta_contrato": ["conta contrato", "contract account", "numero contrato"],
    "instalacao": ["instalacao", "installation", "unidade consumidora", "uc"],
    "leitura_anterior": ["leitura anterior", "previous reading", "leitura ant"],
    "leitura_atual": ["leitura atual", "current reading"],
    "proxima_leitura": ["proxima leitura", "next reading"],
    "mes_ref": ["mes referencia", "mes ref", "reference month", "periodo", "mes ano"],
    "vencimento": ["vencimento", "due date", "data vencimento"],
    "total_fatura": ["total fatura", "total da fatura", "valor total", "total a pagar", "total bill"],
    "nota_fiscal": ["nota fiscal", "numero nota fiscal", "invoice number", "nf"],
    "consumo_kwh": ["consumo kwh", "energia kwh", "consumo do mes", "consumption"],
    "tusd_energia_ponta": ["tusd energia ponta", "tusd ponta", "uso sistema ponta"],
    "tusd_energia_fora_ponta": ["tusd energia fora ponta", "tusd fora ponta", "uso sistema fora ponta"],
    "energia_te": ["energia te", "tarifa de energia", "consumo te"],
    "demanda": ["demanda", "demand", "demanda distrib", "demanda contratada"],
    "demanda_ponta": ["demanda ponta", "dem contratada ponta", "demanda distribuicao ponta"],
    "demanda_fora_ponta": ["demanda fora ponta", "dem contratada fora ponta"],
    "icms": ["icms", "imposto icms", "valor icms"],
    "pis_pasep": ["pis", "pasep", "pis pasep"],
    "cofins": ["cofins", "cofins imposto"],
    "bandeira_tarifaria": ["adicional bandeira", "bandeira amarela", "bandeira vermelha", "adic band"],
    "iluminacao_publica": ["contribuicao iluminacao publica", "cip", "ilum publica", "cosip"],
    "subsidio": ["subsidio tarifario", "subvencao tarifaria", "subsidio"],
    "custo_disponibilidade": ["custo de disponibilidade", "disponibilidade", "custo disp"],
    "multa_juros": ["multa", "juros", "juros moratoria", "correcao monetaria", "multa atraso"],
    "ajuste": ["ajuste", "cobranca ajuste", "ajuste faturamento", "restituicao"],
    "outros": ["outros", "other", "diversos"],
    "cidade": ["cidade", "city", "municipio", "localidade"],
    "endereco": ["endereco", "address", "rua", "logradouro", "avenida"],
    "distribuidora": ["distribuidora", "concessionaria", "ceee", "celesc", "copel", "cpfl", "enel"],
    "cnpj_cpf": ["cnpj", "cpf", "cnpj cpf", "inscricao"],
    "classe_consumo": ["classe", "subclasse", "classe consumo", "tipo de fornecimento", "grupo"],
    "tensao": ["tensao", "tensao contratada", "tensao nominal", "voltagem", "kv"],
}
