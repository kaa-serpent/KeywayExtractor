import fitz  # PyMuPDF

import fitz  # PyMuPDF


def supprimer_pages(pdf_entree, pdf_sortie, nb_premieres_pages=30, nb_derrieres_pages=60):
    # Ouvrir le fichier PDF
    doc = fitz.open(pdf_entree)
    total_pages = len(doc)

    # Vérifier qu'il reste des pages après suppression
    if total_pages <= (nb_premieres_pages + nb_derrieres_pages):
        print("Erreur : Le PDF contient moins de pages que celles à supprimer.")
        return

    # Créer un nouveau document
    new_doc = fitz.open()

    # Ajouter les pages du milieu (sans les 30 premières ni les 59 dernières)
    for i in range(nb_premieres_pages, total_pages - nb_derrieres_pages):
        new_doc.insert_pdf(doc, from_page=i, to_page=i)

    # Sauvegarder le nouveau fichier PDF
    new_doc.save(pdf_sortie)
    new_doc.close()
    doc.close()

    print(f"Le PDF a été enregistré sous '{pdf_sortie}' avec {len(new_doc)} pages restantes.")


supprimer_pages("00-catalogue-110.pdf", "silcaBlanks.pdf")
