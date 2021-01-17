# Copyright 2021 Regeneron Pharmaceuticals Inc. All rights reserved.
# License for Non-Commercial Use of TCRAI code
# All files in this repository (“source code”) are licensed under the following terms below:
# “You” refers to an academic institution or academically employed full-time personnel only. 
# “Regeneron” refers to Regeneron Pharmaceuticals, Inc.
# Regeneron hereby grants You a right to use, reproduce, modify, or distribute the source code to the TCRAI algorithms, in whole or in part, whether in original or modified form, for academic research purposes only.  The foregoing right is royalty-free, worldwide, revocable, non-exclusive, and non-transferable.  
# Prohibited Uses:  The rights granted herein do not include any right to use by commercial entities or commercial use of any kind, including, without limitation, any integration into other code or software that is used for further commercialization, any reproduction, copy, modification or creation of a derivative work that is then incorporated into a commercial product or service or otherwise used for any commercial purpose, or distribution of the source code not in conformity with the restrictions set forth above, whether in whole or in part and whether in original or modified form, and any such commercial usage is not permitted.  
# Except as expressly provided for herein, nothing in this License grants to You any right, title or interest in and to the intellectual property of Regeneron (either expressly or by implication or estoppel).  Notwithstanding anything else in this License, nothing contained herein shall limit or compromise the rights of Regeneron with respect to its own intellectual property or limit its freedom to practice and to develop its products and product candidates.
# If the source code, whole or in part and in original or modified form, is reproduced, shared or distributed in any manner, it must (1) identify Regeneron Pharmaceuticals, Inc. as the original creator, and (2) include the terms of this License.  
# UNLESS OTHERWISE SEPARATELY AGREED UPON, THE SOURCE CODE IS PROVIDED ON AN AS-IS BASIS, AND REGENERON PHARMACEUTICALS, INC. MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND CONCERNING THE SOURCE CODE, IN WHOLE OR IN PART AND IN ORIGINAL OR MODIFIED FORM, WHETHER EXPRESS, IMPLIED, STATUTORY, OR OTHER REPRESENTATIONS OR WARRANTIES. THIS INCLUDES, WITHOUT LIMITATION, WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS, ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT KNOWN OR DISCOVERABLE. 
# In no case shall Regeneron be liable for any loss, claim, damage, or expenses, of any kind, which may arise from or in connection with this License or the use of the source code. You shall indemnify and hold Regeneron and its employees harmless from any loss, claim, damage, expenses, or liability, of any kind, from a third-party which may arise from or in connection with this License or Your use of the source code. 
# You agree that this License and its terms are governed by the laws of the State of New York, without regard to choice of law rules or the United Nations Convention on the International Sale of Goods.
# Please reach out to Regeneron Pharmaceuticals Inc./Administrator relating to any non-academic or commercial use of the source code.
from weblogo import *

def create_logo_pdf(fasta_path,
                   title = None,
                   units='probability'):
    with open(fasta_path, 'r') as f:
        seqs = read_seq_data(f,alphabet=unambiguous_protein_alphabet)
    logodata = LogoData.from_seqs(seqs)
    #print(str(logodata))
    if title is not None:
        logooptions = LogoOptions(logo_title=title)
    else:
        logooptions = LogoOptions()
    logooptions.show_fineprint = False
    logooptions.unit_name = units
    logoformat = LogoFormat(logodata, logooptions)
    if title is not None:
        logoformat.show_title = True
    pdf = pdf_formatter(logodata, logoformat)
    return pdf

def create_logo_png(fasta_path,
                   title = None,
                   units='probability'):
    with open(fasta_path, 'r') as f:
        seqs = read_seq_data(f,alphabet=unambiguous_protein_alphabet)
    logodata = LogoData.from_seqs(seqs)
    #print(str(logodata))
    if title is not None:
        logooptions = LogoOptions(logo_title=title)
    else:
        logooptions = LogoOptions()
    logooptions.show_fineprint = False
    logooptions.unit_name = units
    logoformat = LogoFormat(logodata, logooptions)
    if title is not None:
        logoformat.show_title = True
    png = png_formatter(logodata, logoformat)
    return png

def save_single_logo(logo_pdf_data,out_path):
    with open(out_path, 'wb') as f:
        f.write(logo_pdf_data)
    return None

def save_logo_from_fasta(fasta_path,
                         out_path,
                         title = '',
                         units='probability'):
    pdf = create_logo_pdf(fasta_path,
                   title = title,
                   units=units)
    save_single_logo(pdf,out_path)
    return None