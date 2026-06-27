def retrieve(db, question):

    docs = db.similarity_search(

        question,

        k=4

    )

    context = ""

    for doc in docs:

        context += doc.page_content

        context += "\n\n"

    return context
