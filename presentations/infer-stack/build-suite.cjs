// Build one audience deck or all four; keep the original technical deck intact.
const names=['why','how','evidence','demo'];
async function main(){
    const args=process.argv.slice(2);
    const requested=args.length?args:names;
    for(const name of requested){
        if(!names.includes(name))throw new Error(`Unknown deck ${name}; choose ${names.join(', ')}`);
        await require(`./decks/${name}.cjs`)();
    }
}
main().catch(err=>{console.error(err);process.exitCode=1;});
