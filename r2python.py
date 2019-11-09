from pyensae.languages import r2python

if __name__ == '__main__':
    rscript = """traverse <- function(a,i,innerl){
        if(i < (ncol(df))){
            alevelinner <- as.character(unique(df[which(as.character(df[,i])==a),i+1]))
            desc <- NULL
            if(length(alevelinner) == 1) (newickout <- traverse(alevelinner,i+1,innerl))
            else {
                for(b in alevelinner) desc <- c(desc,traverse(b,i+1,innerl))
                il <- NULL; if(innerl==TRUE) il <- a
                (newickout <- paste("(",paste(desc,collapse=","),")",il,sep=""))
            }
        }
        else { (newickout <- a) }
    }"""

    pyscript1 = print(r2python(rscript, pep8=True))

